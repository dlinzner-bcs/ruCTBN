use crate::ctbn::CTBN;
use crate::dynamics::DynamicsResult;
use ndarray::Array3;

/// Noisy or partial state evidence used by the node-star approximation.
///
/// `likelihoods[node][state]` is the likelihood of the observed value given
/// that node state. Entries may be omitted by using uniform likelihoods.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub time: f64,
    pub likelihoods: Vec<Vec<f64>>,
}

/// Arithmetic and geometric posterior expected conditional rates.
///
/// The paper's mixture approximation distinguishes these for non-degenerate
/// rate mixtures. Each entry has dimensions `[source, target, parent_condition]`
/// and must be a valid conditional intensity matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct StarRateModel {
    pub arithmetic: Vec<Array3<f64>>,
    pub geometric: Vec<Array3<f64>>,
}

impl StarRateModel {
    pub fn degenerate(ctbn: &CTBN) -> Self {
        let arithmetic: Vec<Array3<f64>> =
            ctbn.nodes.iter().map(|node| node.cim.val.clone()).collect();
        Self {
            geometric: arithmetic.clone(),
            arithmetic,
        }
    }
}

/// Effective node-local arithmetic and geometric rate trajectories.
struct EffectiveRates {
    arithmetic: Vec<Vec<Vec<f64>>>,
    geometric: Vec<Vec<Vec<f64>>>,
}

/// Complete output of the SSL-CTBN node-star variational iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct StarResult {
    pub dynamics: DynamicsResult,
    pub backward_responses: Vec<Vec<Vec<f64>>>,
    /// `dwell_times[node][parent_condition][state]`
    pub dwell_times: Vec<Vec<Vec<f64>>>,
    /// `transitions[node][parent_condition][source][target]`
    pub transitions: Vec<Vec<Vec<Vec<f64>>>>,
}

impl StarResult {
    pub fn final_state_probability(&self, state: &[usize]) -> Result<f64, String> {
        self.dynamics.final_state_probability(state)
    }
}

/// Configuration for the SSL-CTBN node-star variational method.
#[derive(Clone, Debug, PartialEq)]
pub struct StarConfig {
    pub iterations: usize,
    pub tolerance: f64,
}

impl Default for StarConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            tolerance: 1e-8,
        }
    }
}

/// Runs the MATLAB SSL-CTBN star variational forward/backward approximation.
///
/// This is distinct from disjoint cluster mean-field inference. The method
/// maintains one singleton marginal trajectory and one backward response per
/// node. Parent assignments are averaged using products of singleton parent
/// marginals, while child effects enter through the backward coupling term.
pub fn star_variational_dynamics(
    ctbn: &CTBN,
    initial_state: &[usize],
    observations: &[Observation],
    duration: f64,
    time_step: f64,
    config: &StarConfig,
) -> Result<StarResult, String> {
    star_variational_dynamics_with_rates(
        ctbn,
        &StarRateModel::degenerate(ctbn),
        initial_state,
        observations,
        duration,
        time_step,
        config,
    )
}

/// Runs node-star inference with explicit arithmetic/geometric posterior rates.
/// Use this entry point for SSL mixture/posterior rates; fixed CIMs use the
/// convenience function above, for which both rate means coincide.
pub fn star_variational_dynamics_with_rates(
    ctbn: &CTBN,
    rate_model: &StarRateModel,
    initial_state: &[usize],
    observations: &[Observation],
    duration: f64,
    time_step: f64,
    config: &StarConfig,
) -> Result<StarResult, String> {
    validate_input(
        ctbn,
        rate_model,
        initial_state,
        observations,
        duration,
        time_step,
        config,
    )?;
    let mut times = Vec::new();
    if duration == 0.0 {
        times.push(0.0);
    } else {
        let steps = (duration / time_step).ceil() as usize;
        times.extend((0..=steps).map(|step| step as f64 * duration / steps as f64));
        times.extend(observations.iter().map(|observation| observation.time));
        times.sort_by(|left, right| left.partial_cmp(right).unwrap());
        times.dedup_by(|left, right| (*left - *right).abs() < 1e-12);
    }
    let steps = times.len() - 1;

    let mut mu = initialize_mu(ctbn, initial_state, &times);
    let mut rho = initialize_rho(ctbn, steps + 1);

    for iteration in 0..config.iterations {
        let previous_mu = mu.clone();
        let previous_rho = rho.clone();
        let mut next_mu = previous_mu.clone();
        let mut next_rho = previous_rho.clone();

        for node_index in 0..ctbn.nodes.len() {
            let rates = effective_rates(ctbn, rate_model, node_index, &previous_mu);
            let response = child_response(
                ctbn,
                rate_model,
                node_index,
                &previous_mu,
                &previous_rho,
                iteration != 0,
            );
            next_rho[node_index] =
                backward_response(ctbn, node_index, &rates, &response, observations, &times);
            next_mu[node_index] = forward_marginal(
                ctbn,
                node_index,
                initial_state[node_index],
                &rates,
                &next_rho[node_index],
                &times,
            );
        }

        let change = max_probability_change(&mu, &next_mu, &rho, &next_rho);
        mu = next_mu;
        rho = next_rho;
        if change <= config.tolerance {
            break;
        }
    }

    let marginals = (0..=steps)
        .map(|step| mu.iter().map(|node| node[step].clone()).collect())
        .collect();
    let clusters: Vec<Vec<usize>> = (0..ctbn.nodes.len()).map(|node| vec![node]).collect();
    let cluster_probabilities = (0..=steps)
        .map(|step| mu.iter().map(|node| node[step].clone()).collect())
        .collect();

    let (dwell_times, transitions) = sufficient_statistics(ctbn, rate_model, &mu, &rho, &times);
    Ok(StarResult {
        dynamics: DynamicsResult {
            times,
            marginals,
            cluster_probabilities,
            clusters,
        },
        backward_responses: rho,
        dwell_times,
        transitions,
    })
}

fn validate_input(
    ctbn: &CTBN,
    rate_model: &StarRateModel,
    initial_state: &[usize],
    observations: &[Observation],
    duration: f64,
    time_step: f64,
    config: &StarConfig,
) -> Result<(), String> {
    if ctbn.nodes.is_empty() {
        return Err("the CTBN must contain at least one node".into());
    }
    if rate_model.arithmetic.len() != ctbn.nodes.len()
        || rate_model.geometric.len() != ctbn.nodes.len()
    {
        return Err("the arithmetic and geometric rate models must contain every node".into());
    }
    for (node_index, node) in ctbn.nodes.iter().enumerate() {
        let expected_shape = (node.d, node.d, node.cim.p);
        if rate_model.arithmetic[node_index].dim() != expected_shape
            || rate_model.geometric[node_index].dim() != expected_shape
        {
            return Err(format!(
                "expected rate model for node {} has shape {:?}, but both means must be {:?}",
                node_index,
                rate_model.arithmetic[node_index].dim(),
                expected_shape
            ));
        }
        for rates in [
            &rate_model.arithmetic[node_index],
            &rate_model.geometric[node_index],
        ] {
            for condition in 0..node.cim.p {
                for source in 0..node.d {
                    for target in 0..node.d {
                        let value = rates[[source, target, condition]];
                        if !value.is_finite()
                            || (source != target && value < 0.0)
                            || (source == target && value > 0.0)
                        {
                            return Err("expected rates must form conditional generators".into());
                        }
                    }
                }
            }
        }
    }
    if initial_state.len() != ctbn.nodes.len() {
        return Err("initial state must contain one state per node".into());
    }
    for (node, &state) in ctbn.nodes.iter().zip(initial_state) {
        if state >= node.d {
            return Err(format!(
                "initial state {} is invalid for node {}",
                state,
                node.index()
            ));
        }
    }
    if !duration.is_finite() || duration < 0.0 {
        return Err("duration must be finite and non-negative".into());
    }
    if !time_step.is_finite() || time_step <= 0.0 {
        return Err("time step must be finite and positive".into());
    }
    if config.iterations == 0 {
        return Err("star iteration count must be positive".into());
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err("star tolerance must be finite and non-negative".into());
    }
    for observation in observations {
        if !observation.time.is_finite() || observation.time < 0.0 || observation.time > duration {
            return Err("observation time must lie within the simulation interval".into());
        }
        if observation.likelihoods.len() != ctbn.nodes.len() {
            return Err("each observation must contain one likelihood per node".into());
        }
        for (node, likelihood) in ctbn.nodes.iter().zip(&observation.likelihoods) {
            if likelihood.len() != node.d {
                return Err(format!(
                    "observation for node {} must contain {} state likelihoods",
                    node.index(),
                    node.d
                ));
            }
            if likelihood
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
                || likelihood.iter().sum::<f64>() <= 0.0
            {
                return Err(
                    "observation likelihoods must be non-negative with positive mass".into(),
                );
            }
        }
    }
    Ok(())
}

fn initialize_mu(ctbn: &CTBN, initial_state: &[usize], times: &[f64]) -> Vec<Vec<Vec<f64>>> {
    let mut output = Vec::with_capacity(ctbn.nodes.len());
    for (node_index, node) in ctbn.nodes.iter().enumerate() {
        let first_rates = node
            .cim
            .val
            .slice(ndarray::s![.., .., 0])
            .to_owned()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let mut trajectory = vec![initial_vector(node.d, initial_state[node_index])];
        for window in 1..times.len() {
            let step = times[window] - times[window - 1];
            let previous = trajectory[window - 1].clone();
            trajectory.push(rk4_forward(&previous, step, &first_rates, None));
        }
        normalize_vector(trajectory.last_mut().unwrap());
        output.push(trajectory);
    }
    output
}

fn initialize_rho(ctbn: &CTBN, count: usize) -> Vec<Vec<Vec<f64>>> {
    ctbn.nodes
        .iter()
        .map(|node| vec![vec![1.0; node.d]; count])
        .collect()
}

fn effective_rates(
    ctbn: &CTBN,
    rate_model: &StarRateModel,
    node_index: usize,
    mu: &[Vec<Vec<f64>>],
) -> EffectiveRates {
    let node = &ctbn.nodes[node_index];
    let mut arithmetic = vec![vec![vec![0.0; node.d]; node.d]; mu[0].len()];
    let mut geometric = vec![vec![vec![0.0; node.d]; node.d]; mu[0].len()];
    for time in 0..mu[0].len() {
        enumerate_conditions(node, |condition, parent_states| {
            let probability: f64 = node
                .parents()
                .iter()
                .zip(parent_states)
                .map(|(&parent, &state)| mu[parent][time][state])
                .product();
            for from in 0..node.d {
                for to in 0..node.d {
                    if from != to {
                        arithmetic[time][from][to] +=
                            probability * rate_model.arithmetic[node_index][[from, to, condition]];
                        geometric[time][from][to] +=
                            probability * rate_model.geometric[node_index][[from, to, condition]];
                    }
                }
            }
        });
        for from in 0..node.d {
            for matrix in [&mut arithmetic[time], &mut geometric[time]] {
                matrix[from][from] = -matrix[from]
                    .iter()
                    .enumerate()
                    .filter(|(to, _)| *to != from)
                    .map(|(_, value)| *value)
                    .sum::<f64>();
            }
        }
    }
    EffectiveRates {
        arithmetic,
        geometric,
    }
}

fn child_response(
    ctbn: &CTBN,
    rate_model: &StarRateModel,
    parent_index: usize,
    mu: &[Vec<Vec<f64>>],
    rho: &[Vec<Vec<f64>>],
    enabled: bool,
) -> Vec<Vec<f64>> {
    let parent_dimension = ctbn.nodes[parent_index].d;
    let mut response = vec![vec![0.0; parent_dimension]; mu[0].len()];
    if !enabled {
        return response;
    }

    for (child_index, child) in ctbn.nodes.iter().enumerate() {
        let Some(parent_position) = child
            .parents()
            .iter()
            .position(|&node| node == parent_index)
        else {
            continue;
        };
        for time in 0..mu[0].len() {
            for parent_state in 0..parent_dimension {
                let mut arithmetic_rates = vec![vec![0.0; child.d]; child.d];
                let mut geometric_rates = vec![vec![0.0; child.d]; child.d];
                enumerate_conditions(child, |condition, parent_states| {
                    if parent_states[parent_position] != parent_state {
                        return;
                    }
                    let probability: f64 = child
                        .parents()
                        .iter()
                        .enumerate()
                        .filter(|(position, _)| *position != parent_position)
                        .map(|(position, &parent)| mu[parent][time][parent_states[position]])
                        .product();
                    for from in 0..child.d {
                        for to in 0..child.d {
                            if from != to {
                                arithmetic_rates[from][to] += probability
                                    * rate_model.arithmetic[child_index][[from, to, condition]];
                                geometric_rates[from][to] += probability
                                    * rate_model.geometric[child_index][[from, to, condition]];
                            }
                        }
                    }
                });
                for from in 0..child.d {
                    for to in 0..child.d {
                        if from == to {
                            continue;
                        }
                        let source_rho = rho[child_index][time][from].max(1e-12);
                        response[time][parent_state] += mu[child_index][time][from]
                            * (geometric_rates[from][to] * rho[child_index][time][to] / source_rho
                                - arithmetic_rates[from][to]);
                    }
                }
            }
        }
    }
    response
}

fn backward_response(
    ctbn: &CTBN,
    node_index: usize,
    rates: &EffectiveRates,
    response: &[Vec<f64>],
    observations: &[Observation],
    times: &[f64],
) -> Vec<Vec<f64>> {
    let node = &ctbn.nodes[node_index];
    let last = times.len() - 1;
    let mut trajectory = vec![vec![0.0; node.d]; times.len()];
    trajectory[last] = vec![1.0; node.d];
    apply_observations(&mut trajectory[last], node_index, times[last], observations);

    for right in (1..times.len()).rev() {
        let left = right - 1;
        let step = times[left] - times[right];
        trajectory[left] = rk4_backward(&trajectory[right], step, rates, response, right, left);
        apply_observations(&mut trajectory[left], node_index, times[left], observations);
        normalize_vector(&mut trajectory[left]);
    }
    trajectory
}

fn apply_observations(
    value: &mut [f64],
    node_index: usize,
    time: f64,
    observations: &[Observation],
) {
    for observation in observations
        .iter()
        .filter(|observation| (observation.time - time).abs() <= 1e-10)
    {
        for (state, likelihood) in observation.likelihoods[node_index].iter().enumerate() {
            value[state] *= likelihood;
        }
    }
}

fn forward_marginal(
    ctbn: &CTBN,
    node_index: usize,
    initial_state: usize,
    rates: &EffectiveRates,
    rho: &[Vec<f64>],
    times: &[f64],
) -> Vec<Vec<f64>> {
    let node = &ctbn.nodes[node_index];
    let mut trajectory = vec![initial_vector(node.d, initial_state)];
    for right in 1..times.len() {
        let left = right - 1;
        trajectory.push(rk4_tilted_forward(
            &trajectory[left],
            times[right] - times[left],
            &rates.geometric,
            rho,
            left,
            right,
        ));
    }
    trajectory
}

fn rk4_forward(
    value: &[f64],
    step: f64,
    rates: &Vec<Vec<f64>>,
    observations: Option<&[f64]>,
) -> Vec<f64> {
    let derivative = |y: &[f64]| forward_derivative(rates, y);
    let k1 = derivative(value);
    let k2 = derivative(&offset(value, &k1, step * 0.5));
    let k3 = derivative(&offset(value, &k2, step * 0.5));
    let k4 = derivative(&offset(value, &k3, step));
    let mut result = value
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            x + step * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0
        })
        .collect::<Vec<_>>();
    if let Some(observation) = observations {
        for (x, likelihood) in result.iter_mut().zip(observation) {
            *x *= likelihood;
        }
    }
    normalize_vector(&mut result);
    result
}

fn rk4_tilted_forward(
    value: &[f64],
    step: f64,
    rates: &[Vec<Vec<f64>>],
    rho: &[Vec<f64>],
    left: usize,
    right: usize,
) -> Vec<f64> {
    let derivative = |y: &[f64], endpoint: usize| {
        let dimension = y.len();
        let mut result = vec![0.0; dimension];
        for source in 0..dimension {
            let outgoing: f64 = (0..dimension)
                .filter(|&destination| destination != source)
                .map(|destination| {
                    rates[endpoint][source][destination] * rho[endpoint][destination]
                        / rho[endpoint][source].max(1e-12)
                })
                .sum();
            result[source] -= outgoing * y[source];
            for destination in 0..dimension {
                if source != destination {
                    result[destination] += rates[endpoint][source][destination]
                        * rho[endpoint][destination]
                        / rho[endpoint][source].max(1e-12)
                        * y[source];
                }
            }
        }
        result
    };
    let k1 = derivative(value, left);
    let k2 = derivative(&offset(value, &k1, step * 0.5), left);
    let k3 = derivative(&offset(value, &k2, step * 0.5), right);
    let k4 = derivative(&offset(value, &k3, step), right);
    let mut result = value
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            x + step * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0
        })
        .collect::<Vec<_>>();
    normalize_vector(&mut result);
    result
}

fn rk4_backward(
    value: &[f64],
    step: f64,
    rates: &EffectiveRates,
    response: &[Vec<f64>],
    right: usize,
    left: usize,
) -> Vec<f64> {
    let derivative = |endpoint: usize, y: &[f64]| {
        let dimension = y.len();
        let mut backward_matrix = vec![vec![0.0; dimension]; dimension];
        for state in 0..dimension {
            for destination in 0..dimension {
                if state != destination {
                    backward_matrix[state][destination] =
                        rates.geometric[endpoint][state][destination];
                }
            }
            backward_matrix[state][state] =
                rates.arithmetic[endpoint][state][state] + response[endpoint][state];
        }
        (0..dimension)
            .map(|state| {
                -(0..dimension)
                    .map(|other| backward_matrix[state][other] * y[other])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>()
    };
    let k1 = derivative(right, value);
    let k2 = derivative(right, &offset(value, &k1, step * 0.5));
    let k3 = derivative(left, &offset(value, &k2, step * 0.5));
    let k4 = derivative(left, &offset(value, &k3, step));
    value
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            x + step * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0
        })
        .collect()
}

fn forward_derivative(rates: &[Vec<f64>], value: &[f64]) -> Vec<f64> {
    value
        .iter()
        .enumerate()
        .map(|(destination, _)| {
            rates
                .iter()
                .enumerate()
                .map(|(source, row)| {
                    if source == destination {
                        rates[source][destination] * value[source]
                    } else {
                        row[destination] * value[source]
                    }
                })
                .sum()
        })
        .collect()
}

fn enumerate_conditions<F>(node: &crate::ctbn::NODE, mut action: F)
where
    F: FnMut(usize, &[usize]),
{
    let count: usize = node.parents_d.iter().product();
    for condition in 0..count {
        let mut remaining = condition;
        let states: Vec<usize> = node
            .parents_d
            .iter()
            .map(|dimension| {
                let state = remaining % dimension;
                remaining /= dimension;
                state
            })
            .collect();
        action(condition, &states);
    }
}

fn sufficient_statistics(
    ctbn: &CTBN,
    rate_model: &StarRateModel,
    mu: &[Vec<Vec<f64>>],
    rho: &[Vec<Vec<f64>>],
    times: &[f64],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<Vec<f64>>>>) {
    let mut all_dwell = Vec::with_capacity(ctbn.nodes.len());
    let mut all_transitions = Vec::with_capacity(ctbn.nodes.len());

    for (node_index, node) in ctbn.nodes.iter().enumerate() {
        let condition_count: usize = node.parents_d.iter().product();
        let mut dwell = vec![vec![0.0; node.d]; condition_count];
        let mut transitions = vec![vec![vec![0.0; node.d]; node.d]; condition_count];

        for step in 0..times.len().saturating_sub(1) {
            let next = step + 1;
            let half_step = (times[next] - times[step]) * 0.5;
            enumerate_conditions(node, |condition, parent_states| {
                let parent_probability = |time: usize| {
                    node.parents()
                        .iter()
                        .zip(parent_states)
                        .map(|(&parent, &state)| mu[parent][time][state])
                        .product::<f64>()
                };
                let left_parent = parent_probability(step);
                let right_parent = parent_probability(next);
                for source in 0..node.d {
                    let left = mu[node_index][step][source] * left_parent;
                    let right = mu[node_index][next][source] * right_parent;
                    dwell[condition][source] += half_step * (left + right);
                    for destination in 0..node.d {
                        if source == destination {
                            continue;
                        }
                        let left_transition = left
                            * rate_model.geometric[node_index][[source, destination, condition]]
                            * rho[node_index][step][destination]
                            / rho[node_index][step][source].max(1e-12);
                        let right_transition = right
                            * rate_model.geometric[node_index][[source, destination, condition]]
                            * rho[node_index][next][destination]
                            / rho[node_index][next][source].max(1e-12);
                        transitions[condition][source][destination] +=
                            half_step * (left_transition + right_transition);
                    }
                }
            });
        }
        all_dwell.push(dwell);
        all_transitions.push(transitions);
    }

    (all_dwell, all_transitions)
}

fn initial_vector(dimension: usize, state: usize) -> Vec<f64> {
    let mut value = vec![0.0; dimension];
    value[state] = 1.0;
    value
}

fn offset(value: &[f64], derivative: &[f64], scale: f64) -> Vec<f64> {
    value
        .iter()
        .zip(derivative)
        .map(|(x, dx)| x + scale * dx)
        .collect()
}

fn normalize_vector(value: &mut [f64]) {
    let sum: f64 = value.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for item in value {
            *item /= sum;
        }
    }
}

fn max_probability_change(
    mu: &[Vec<Vec<f64>>],
    next_mu: &[Vec<Vec<f64>>],
    rho: &[Vec<Vec<f64>>],
    next_rho: &[Vec<Vec<f64>>],
) -> f64 {
    mu.iter()
        .zip(next_mu)
        .flat_map(|(old_node, new_node)| old_node.iter().zip(new_node))
        .flat_map(|(old_time, new_time)| old_time.iter().zip(new_time))
        .map(|(old, new)| (old - new).abs())
        .chain(
            rho.iter()
                .zip(next_rho)
                .flat_map(|(old_node, new_node)| old_node.iter().zip(new_node))
                .flat_map(|(old_time, new_time)| old_time.iter().zip(new_time))
                .map(|(old, new)| ((old / new.max(1e-12)) - 1.0).abs()),
        )
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn single_node_ctbn() -> CTBN {
        let mut rates = Array3::zeros((2, 2, 1));
        rates[[0, 0, 0]] = -1.0;
        rates[[0, 1, 0]] = 1.0;
        rates[[1, 0, 0]] = 2.0;
        rates[[1, 1, 0]] = -2.0;
        CTBN::create_ctbn_from_rates(&[vec![]], &[2], &[rates])
    }

    #[test]
    fn star_without_evidence_matches_exact_forward_dynamics() {
        let ctbn = single_node_ctbn();
        let exact = crate::dynamics::exact_dynamics(&ctbn, &[0], 1.0, 0.001).unwrap();
        let star = star_variational_dynamics(&ctbn, &[0], &[], 1.0, 0.001, &StarConfig::default())
            .unwrap();
        assert!(
            (star.dynamics.marginals.last().unwrap()[0][1] - exact.marginals.last().unwrap()[0][1])
                .abs()
                < 2e-6
        );
        let total_dwell: f64 = star.dwell_times[0][0].iter().sum();
        assert!((total_dwell - 1.0).abs() < 2e-6);
        assert!(star.transitions[0][0][0][1] > 0.0);
    }

    #[test]
    fn non_degenerate_posterior_uses_geometric_rates_for_forward_flow() {
        let ctbn = single_node_ctbn();
        let arithmetic = ctbn.nodes[0].cim.val.clone();
        let mut geometric = Array3::zeros((2, 2, 1));
        geometric[[0, 0, 0]] = -0.5;
        geometric[[0, 1, 0]] = 0.5;
        geometric[[1, 0, 0]] = 1.0;
        geometric[[1, 1, 0]] = -1.0;
        let rates = StarRateModel {
            arithmetic: vec![arithmetic],
            geometric: vec![geometric],
        };
        let result = star_variational_dynamics_with_rates(
            &ctbn,
            &rates,
            &[0],
            &[],
            1.0,
            0.001,
            &StarConfig::default(),
        )
        .unwrap();
        let probability = result.dynamics.marginals.last().unwrap()[0][1];
        let ordinary_geometric_generator = (1.0 - (-1.5_f64).exp()) / 3.0;
        assert!(
            (probability - ordinary_geometric_generator).abs() > 0.02,
            "node-star did not use the distinct arithmetic backward mean: {}",
            probability
        );
        assert!(probability > 0.0 && probability < 1.0);
    }

    #[test]
    fn terminal_evidence_reweights_single_node_path() {
        let ctbn = single_node_ctbn();
        let observation = Observation {
            time: 1.0,
            likelihoods: vec![vec![0.01, 0.99]],
        };
        let star = star_variational_dynamics(
            &ctbn,
            &[0],
            &[observation],
            1.0,
            0.001,
            &StarConfig::default(),
        )
        .unwrap();
        let exact = crate::dynamics::exact_dynamics(&ctbn, &[0], 1.0, 0.001).unwrap();
        let prior = exact.marginals.last().unwrap()[0].clone();
        let exact_posterior = prior[1] * 0.99 / (prior[0] * 0.01 + prior[1] * 0.99);
        let final_probability = star.dynamics.marginals.last().unwrap()[0][1];
        assert!(
            (final_probability - exact_posterior).abs() < 3e-5,
            "terminal probability was {}, exact posterior is {}",
            final_probability,
            exact_posterior
        );
    }
}
