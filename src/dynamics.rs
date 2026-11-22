use crate::common::convert2dec;
use crate::ctbn::{CTBN, NODE};

/// Time-indexed output of numerical CTBN dynamics.
///
/// `marginals[time][node][state]` contains node-state probabilities, while
/// `cluster_probabilities[time][cluster][state]` contains joint cluster beliefs.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicsResult {
    pub times: Vec<f64>,
    pub marginals: Vec<Vec<Vec<f64>>>,
    pub cluster_probabilities: Vec<Vec<Vec<f64>>>,
    pub clusters: Vec<Vec<usize>>,
}

/// Propagates the full CTBN master equation with fourth-order Runge-Kutta.
pub fn exact_dynamics(
    ctbn: &CTBN,
    initial_state: &[usize],
    duration: f64,
    time_step: f64,
) -> Result<DynamicsResult, String> {
    cluster_variational_dynamics(
        ctbn,
        initial_state,
        &[(0..ctbn.nodes.len()).collect()],
        duration,
        time_step,
    )
}

/// Propagates a disjoint-cluster mean-field approximation.
///
/// Transitions inside a cluster use its joint belief, while rates depending on
/// external parent clusters are averaged using those joint beliefs. Different
/// external clusters are assumed independent. Singleton clusters therefore
/// reduce to the star/mean-field approximation used by the CVM MATLAB code in
/// `ssl-ctbn`; one cluster containing all nodes is numerically exact.
pub fn cluster_variational_dynamics(
    ctbn: &CTBN,
    initial_state: &[usize],
    clusters: &[Vec<usize>],
    duration: f64,
    time_step: f64,
) -> Result<DynamicsResult, String> {
    validate_input(ctbn, initial_state, clusters, duration, time_step)?;

    let steps = ((duration / time_step).ceil() as usize).max(1);
    let dt = duration / steps as f64;
    let mut probabilities = initial_cluster_probabilities(ctbn, initial_state, clusters);
    let mut result = DynamicsResult {
        times: vec![0.0],
        marginals: vec![node_marginals(ctbn, clusters, &probabilities)],
        cluster_probabilities: vec![probabilities.clone()],
        clusters: clusters.to_vec(),
    };
    if duration == 0.0 {
        return Ok(result);
    }

    for step in 1..=steps {
        let k1 = derivative(ctbn, clusters, &probabilities);
        let stage2 = combine(&probabilities, &k1, dt * 0.5);
        let k2 = derivative(ctbn, clusters, &stage2);
        let stage3 = combine(&probabilities, &k2, dt * 0.5);
        let k3 = derivative(ctbn, clusters, &stage3);
        let stage4 = combine(&probabilities, &k3, dt);
        let k4 = derivative(ctbn, clusters, &stage4);

        probabilities = probabilities
            .iter()
            .enumerate()
            .map(|(cluster, probability)| {
                probability
                    .iter()
                    .enumerate()
                    .map(|(state, &value)| {
                        value
                            + dt * (k1[cluster][state]
                                + 2.0 * k2[cluster][state]
                                + 2.0 * k3[cluster][state]
                                + k4[cluster][state])
                                / 6.0
                    })
                    .collect()
            })
            .collect();
        normalize_all(&mut probabilities);

        result.times.push(step as f64 * dt);
        result
            .marginals
            .push(node_marginals(ctbn, clusters, &probabilities));
        result.cluster_probabilities.push(probabilities.clone());
    }

    Ok(result)
}

fn validate_input(
    ctbn: &CTBN,
    initial_state: &[usize],
    clusters: &[Vec<usize>],
    duration: f64,
    time_step: f64,
) -> Result<(), String> {
    if ctbn.nodes.is_empty() {
        return Err("the CTBN must contain at least one node".into());
    }
    if duration < 0.0 || !duration.is_finite() {
        return Err("duration must be a finite, non-negative value".into());
    }
    if time_step <= 0.0 || !time_step.is_finite() {
        return Err("time step must be a finite, positive value".into());
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
    if clusters.is_empty() || clusters.iter().any(Vec::is_empty) {
        return Err("clusters must be non-empty".into());
    }
    let mut assigned = vec![false; ctbn.nodes.len()];
    for cluster in clusters {
        for &node in cluster {
            if node >= ctbn.nodes.len() {
                return Err(format!("cluster node {} is outside the CTBN", node));
            }
            if assigned[node] {
                return Err("clusters must be disjoint and cover every node exactly once".into());
            }
            assigned[node] = true;
        }
    }
    if assigned.iter().any(|&value| !value) {
        return Err("clusters must cover every CTBN node".into());
    }
    Ok(())
}

fn initial_cluster_probabilities(
    ctbn: &CTBN,
    initial_state: &[usize],
    clusters: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    clusters
        .iter()
        .map(|cluster| {
            let dimensions: Vec<usize> = cluster.iter().map(|&node| ctbn.nodes[node].d).collect();
            let local_state: Vec<usize> = cluster.iter().map(|&node| initial_state[node]).collect();
            let mut probability = vec![0.0; dimensions.iter().product()];
            probability[convert2dec(&local_state, &dimensions)] = 1.0;
            probability
        })
        .collect()
}

fn derivative(ctbn: &CTBN, clusters: &[Vec<usize>], probabilities: &[Vec<f64>]) -> Vec<Vec<f64>> {
    clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| {
            let dimensions: Vec<usize> = cluster.iter().map(|&node| ctbn.nodes[node].d).collect();
            let mut result = vec![0.0; dimensions.iter().product()];

            for (source_index, &source_probability) in
                probabilities[cluster_index].iter().enumerate()
            {
                let source_state = decode_index(source_index, &dimensions);
                for (local_node, &node_index) in cluster.iter().enumerate() {
                    let node = &ctbn.nodes[node_index];
                    for target in 0..node.d {
                        if target == source_state[local_node] {
                            continue;
                        }
                        let rate = expected_transition_rate(
                            ctbn,
                            node,
                            source_state[local_node],
                            target,
                            cluster_index,
                            cluster,
                            &source_state,
                            clusters,
                            probabilities,
                        );
                        let flow = source_probability * rate;
                        result[source_index] -= flow;

                        let mut target_state = source_state.clone();
                        target_state[local_node] = target;
                        result[convert2dec(&target_state, &dimensions)] += flow;
                    }
                }
            }
            result
        })
        .collect()
}

fn expected_transition_rate(
    ctbn: &CTBN,
    node: &NODE,
    from: usize,
    to: usize,
    local_cluster_index: usize,
    local_cluster: &[usize],
    local_state: &[usize],
    clusters: &[Vec<usize>],
    probabilities: &[Vec<f64>],
) -> f64 {
    type ParentAssignments = Vec<(usize, usize)>;

    fn visit_external_clusters(
        node: &NODE,
        ctbn: &CTBN,
        from: usize,
        to: usize,
        group_position: usize,
        condition: usize,
        probability: f64,
        external_clusters: &[(usize, ParentAssignments)],
        clusters: &[Vec<usize>],
        probabilities: &[Vec<f64>],
    ) -> f64 {
        if group_position == external_clusters.len() {
            return probability * node.transition_rate(from, to, condition);
        }

        let (cluster_index, assignments) = &external_clusters[group_position];
        let dimensions: Vec<usize> = clusters[*cluster_index]
            .iter()
            .map(|&node| ctbn.nodes[node].d)
            .collect();
        probabilities[*cluster_index]
            .iter()
            .enumerate()
            .map(|(state_index, &cluster_probability)| {
                let cluster_state = decode_index(state_index, &dimensions);
                let added_condition: usize = assignments
                    .iter()
                    .map(|&(local_position, stride)| cluster_state[local_position] * stride)
                    .sum();
                visit_external_clusters(
                    node,
                    ctbn,
                    from,
                    to,
                    group_position + 1,
                    condition + added_condition,
                    probability * cluster_probability,
                    external_clusters,
                    clusters,
                    probabilities,
                )
            })
            .sum()
    }

    let mut strides = Vec::with_capacity(node.parents().len());
    let mut stride = 1;
    for dimension in &node.parents_d {
        strides.push(stride);
        stride *= dimension;
    }

    let mut condition = 0;
    let mut external_clusters: Vec<(usize, ParentAssignments)> = Vec::new();
    for (parent_position, &parent) in node.parents().iter().enumerate() {
        if let Some(local_position) = local_cluster.iter().position(|&node| node == parent) {
            condition += local_state[local_position] * strides[parent_position];
            continue;
        }

        let (cluster_index, local_position) = clusters
            .iter()
            .enumerate()
            .find_map(|(index, cluster)| {
                cluster
                    .iter()
                    .position(|&node| node == parent)
                    .map(|position| (index, position))
            })
            .expect("all nodes must belong to a cluster");
        assert_ne!(cluster_index, local_cluster_index);
        match external_clusters
            .iter_mut()
            .find(|(index, _)| *index == cluster_index)
        {
            Some((_, assignments)) => {
                assignments.push((local_position, strides[parent_position]));
            }
            None => external_clusters.push((
                cluster_index,
                vec![(local_position, strides[parent_position])],
            )),
        }
    }

    visit_external_clusters(
        node,
        ctbn,
        from,
        to,
        0,
        condition,
        1.0,
        &external_clusters,
        clusters,
        probabilities,
    )
}

fn node_marginals(
    ctbn: &CTBN,
    clusters: &[Vec<usize>],
    probabilities: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let mut marginals: Vec<Vec<f64>> = ctbn.nodes.iter().map(|node| vec![0.0; node.d]).collect();

    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let dimensions: Vec<usize> = cluster.iter().map(|&node| ctbn.nodes[node].d).collect();
        for (state_index, &probability) in probabilities[cluster_index].iter().enumerate() {
            let state = decode_index(state_index, &dimensions);
            for (local_position, &node) in cluster.iter().enumerate() {
                marginals[node][state[local_position]] += probability;
            }
        }
    }
    marginals
}

fn decode_index(mut index: usize, dimensions: &[usize]) -> Vec<usize> {
    dimensions
        .iter()
        .map(|&dimension| {
            let state = index % dimension;
            index /= dimension;
            state
        })
        .collect()
}

fn combine(values: &[Vec<f64>], derivatives: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    values
        .iter()
        .zip(derivatives)
        .map(|(value, derivative)| {
            value
                .iter()
                .zip(derivative)
                .map(|(&x, &dx)| x + scale * dx)
                .collect()
        })
        .collect()
}

fn normalize_all(probabilities: &mut [Vec<f64>]) {
    for probability in probabilities {
        for value in probability.iter_mut() {
            if *value < 0.0 && *value > -1e-12 {
                *value = 0.0;
            }
        }
        let sum: f64 = probability.iter().sum();
        assert!(sum.is_finite() && sum > 0.0, "invalid dynamics probability");
        for value in probability {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Sampler;
    use ndarray::Array3;
    use rand::{SeedableRng, StdRng};

    fn test_ctbn() -> CTBN {
        let mut node0 = Array3::zeros((2, 2, 1));
        node0[[0, 0, 0]] = -1.0;
        node0[[0, 1, 0]] = 1.0;
        node0[[1, 0, 0]] = 2.0;
        node0[[1, 1, 0]] = -2.0;

        let mut node1 = Array3::zeros((2, 2, 2));
        node1[[0, 0, 0]] = -0.5;
        node1[[0, 1, 0]] = 0.5;
        node1[[1, 0, 0]] = 0.5;
        node1[[1, 1, 0]] = -0.5;
        node1[[0, 0, 1]] = -3.0;
        node1[[0, 1, 1]] = 3.0;
        node1[[1, 0, 1]] = 0.2;
        node1[[1, 1, 1]] = -0.2;

        CTBN::create_ctbn_from_rates(&[vec![], vec![0]], &[2, 2], &[node0, node1])
    }

    #[test]
    fn exact_dynamics_matches_the_two_state_analytic_solution() {
        let mut rates = Array3::zeros((2, 2, 1));
        rates[[0, 0, 0]] = -1.0;
        rates[[0, 1, 0]] = 1.0;
        rates[[1, 0, 0]] = 2.0;
        rates[[1, 1, 0]] = -2.0;
        let ctbn = CTBN::create_ctbn_from_rates(&[vec![]], &[2], &[rates]);

        let result = exact_dynamics(&ctbn, &[0], 1.0, 0.001).unwrap();
        let p_one = result.marginals.last().unwrap()[0][1];
        let analytic = 1.0 / 3.0 * (1.0 - (-3.0_f64).exp());
        assert!((p_one - analytic).abs() < 2e-7, "{} != {}", p_one, analytic);
    }

    #[test]
    fn exact_dynamics_agrees_with_direct_sampling() {
        let ctbn = test_ctbn();
        let duration = 0.7;
        let exact = exact_dynamics(&ctbn, &[0, 0], duration, 0.002).unwrap();
        let expected = &exact.marginals.last().unwrap();

        let trials = 20_000;
        let mut rng = StdRng::from_seed([23; 32]);
        let mut counts = vec![vec![0usize; 2]; 2];
        for _ in 0..trials {
            let mut sampler = Sampler::create_sampler(&ctbn, &[0, 0], &duration);
            sampler.sample_path_with_rng(&mut rng);
            let state = sampler.state_at(duration).unwrap();
            for (node, &value) in state.iter().enumerate() {
                counts[node][value] += 1;
            }
        }

        for node in 0..2 {
            for state in 0..2 {
                let sampled = counts[node][state] as f64 / trials as f64;
                assert!(
                    (sampled - expected[node][state]).abs() < 0.02,
                    "node {}, state {}: sampled {}, exact {}",
                    node,
                    state,
                    sampled,
                    expected[node][state]
                );
            }
        }
    }

    #[test]
    fn cluster_approximation_uses_exact_internal_correlations() {
        let ctbn = test_ctbn();
        let exact = exact_dynamics(&ctbn, &[0, 0], 1.0, 0.002).unwrap();
        let mean_field =
            cluster_variational_dynamics(&ctbn, &[0, 0], &[vec![0], vec![1]], 1.0, 0.002).unwrap();

        for (clustered, expected) in mean_field
            .marginals
            .last()
            .unwrap()
            .iter()
            .zip(exact.marginals.last().unwrap())
        {
            for (&approximation, &exact_value) in clustered.iter().zip(expected) {
                assert!((approximation - exact_value).abs() < 0.12);
            }
        }
        assert!(mean_field
            .cluster_probabilities
            .iter()
            .flat_map(|step| step.iter())
            .flat_map(|cluster| cluster.iter())
            .all(|value| value.is_finite() && *value >= 0.0));
    }
}
