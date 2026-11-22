use ctbn::ctbn::CTBN;
use ctbn::dynamics::{cluster_variational_dynamics, exact_dynamics};
use ctbn::sampler::Sampler;
use ctbn::star::{star_variational_dynamics, Observation, StarConfig};
use ndarray::Array3;
use rand::{SeedableRng, StdRng};
use std::hint::black_box;
use std::time::{Duration, Instant};

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

fn independent_rates(forward: f64, backward: f64) -> Array3<f64> {
    let mut rates = Array3::zeros((2, 2, 1));
    rates[[0, 0, 0]] = -forward;
    rates[[0, 1, 0]] = forward;
    rates[[1, 0, 0]] = backward;
    rates[[1, 1, 0]] = -backward;
    rates
}

fn conditional_rates(values: &[(f64, f64)]) -> Array3<f64> {
    let mut rates = Array3::zeros((2, 2, values.len()));
    for (condition, &(forward, backward)) in values.iter().enumerate() {
        rates[[0, 0, condition]] = -forward;
        rates[[0, 1, condition]] = forward;
        rates[[1, 0, condition]] = backward;
        rates[[1, 1, condition]] = -backward;
    }
    rates
}

fn six_node_ctbn() -> CTBN {
    CTBN::create_ctbn_from_rates(
        &[vec![5], vec![], vec![0, 1], vec![], vec![], vec![3, 4, 2]],
        &[2, 2, 2, 2, 2, 2],
        &[
            conditional_rates(&[(0.9, 2.0), (2.2, 0.7)]),
            independent_rates(0.7, 1.3),
            conditional_rates(&[(0.4, 0.8), (2.8, 0.3), (1.4, 0.5), (3.5, 0.2)]),
            independent_rates(0.9, 1.8),
            independent_rates(1.2, 0.6),
            conditional_rates(&[
                (0.3, 0.9),
                (1.1, 0.5),
                (2.0, 0.4),
                (3.4, 0.2),
                (0.8, 0.7),
                (1.7, 0.4),
                (2.6, 0.2),
                (4.1, 0.1),
            ]),
        ],
    )
}

fn decode_state(mut index: usize, dimensions: &[usize]) -> Vec<usize> {
    dimensions
        .iter()
        .map(|&dimension| {
            let state = index % dimension;
            index /= dimension;
            state
        })
        .collect()
}

fn exact_terminal_posterior(
    exact: &ctbn::dynamics::DynamicsResult,
    likelihoods: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let dimensions: Vec<usize> = exact
        .marginals
        .last()
        .unwrap()
        .iter()
        .map(Vec::len)
        .collect();
    let joint = &exact.cluster_probabilities.last().unwrap()[0];
    let mut posterior = vec![vec![0.0; 2]; dimensions.len()];
    let mut evidence = 0.0;
    for (index, &probability) in joint.iter().enumerate() {
        let state = decode_state(index, &dimensions);
        let weight: f64 = state
            .iter()
            .zip(likelihoods)
            .map(|(&state, likelihood)| likelihood[state])
            .product::<f64>()
            * probability;
        evidence += weight;
        for (node, &node_state) in state.iter().enumerate() {
            posterior[node][node_state] += weight;
        }
    }
    for node in &mut posterior {
        for value in node {
            *value /= evidence;
        }
    }
    posterior
}

fn independent_terminal_posterior(prior: &[Vec<f64>], likelihoods: &[Vec<f64>]) -> Vec<Vec<f64>> {
    prior
        .iter()
        .zip(likelihoods)
        .map(|(marginal, likelihood)| {
            let unnormalized: Vec<f64> = marginal
                .iter()
                .zip(likelihood)
                .map(|(probability, likelihood)| probability * likelihood)
                .collect();
            let total: f64 = unnormalized.iter().sum();
            unnormalized
                .into_iter()
                .map(|value| value / total)
                .collect()
        })
        .collect()
}

fn average(elapsed: Duration, iterations: usize) -> Duration {
    elapsed / iterations as u32
}

fn main() {
    const DURATION: f64 = 0.7;
    const STEP: f64 = 0.002;
    const INFERENCE_RUNS: usize = 100;
    const SAMPLE_RUNS: usize = 20_000;
    let ctbn = test_ctbn();

    let started = Instant::now();
    for _ in 0..INFERENCE_RUNS {
        black_box(exact_dynamics(&ctbn, &[0, 0], DURATION, STEP).unwrap());
    }
    let exact = average(started.elapsed(), INFERENCE_RUNS);

    let started = Instant::now();
    for _ in 0..INFERENCE_RUNS {
        black_box(
            cluster_variational_dynamics(&ctbn, &[0, 0], &[vec![0], vec![1]], DURATION, STEP)
                .unwrap(),
        );
    }
    let cluster = average(started.elapsed(), INFERENCE_RUNS);

    let started = Instant::now();
    let mut rng = StdRng::from_seed([23; 32]);
    let mut final_state_count = 0;
    for _ in 0..SAMPLE_RUNS {
        let mut sampler = Sampler::create_sampler(&ctbn, &[0, 0], &DURATION);
        sampler.sample_path_with_rng(&mut rng);
        final_state_count += sampler.state_at(DURATION).unwrap()[1];
    }
    black_box(final_state_count);
    let sampling = average(started.elapsed(), SAMPLE_RUNS);

    let exact_result = exact_dynamics(&ctbn, &[0, 0], DURATION, STEP).unwrap();
    let cluster_result =
        cluster_variational_dynamics(&ctbn, &[0, 0], &[vec![0], vec![1]], DURATION, STEP).unwrap();
    let exact_final = exact_result.marginals.last().unwrap();
    let cluster_final = cluster_result.marginals.last().unwrap();
    let max_error = exact_final
        .iter()
        .zip(cluster_final)
        .flat_map(|(exact_node, cluster_node)| {
            exact_node
                .iter()
                .zip(cluster_node)
                .map(|(exact_value, cluster_value)| (exact_value - cluster_value).abs())
        })
        .fold(0.0, f64::max);

    println!("workload: 2 binary nodes, duration={DURATION}, time_step={STEP}");
    println!("exact inference:     {:?} average/run", exact);
    println!("cluster variational: {:?} average/run", cluster);
    println!("sampling:            {:?} average/path", sampling);
    println!(
        "node 1 P(state=1): exact {:.6}, CVM {:.6}, sampling {:.6}",
        exact_final[1][1],
        cluster_final[1][1],
        final_state_count as f64 / SAMPLE_RUNS as f64
    );
    println!("CVM maximum marginal error: {:.6}", max_error);

    let ctbn = six_node_ctbn();
    let initial = [0, 0, 0, 0, 0, 0];
    let likelihoods = vec![
        vec![0.95, 0.05],
        vec![0.05, 0.95],
        vec![0.90, 0.10],
        vec![0.10, 0.90],
        vec![0.85, 0.15],
        vec![0.15, 0.85],
    ];
    let observations = vec![Observation {
        time: DURATION,
        likelihoods: likelihoods.clone(),
    }];
    let exact = exact_dynamics(&ctbn, &initial, DURATION, STEP).unwrap();
    let exact_posterior = exact_terminal_posterior(&exact, &likelihoods);
    let singleton_prior = exact.marginals.last().unwrap();
    let singleton_posterior = independent_terminal_posterior(singleton_prior, &likelihoods);

    let started = Instant::now();
    let star = star_variational_dynamics(
        &ctbn,
        &initial,
        &observations,
        DURATION,
        STEP,
        &StarConfig::default(),
    )
    .unwrap();
    let star_time = started.elapsed();
    let star_posterior = star.dynamics.marginals.last().unwrap();
    let max_error = |approximation: &[Vec<f64>]| {
        approximation
            .iter()
            .zip(&exact_posterior)
            .flat_map(|(actual, expected)| {
                actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
            })
            .fold(0.0, f64::max)
    };

    println!();
    println!("conditioned 6-node benchmark, same fixed CIM for every solver");
    println!("MATLAB node-star: {:?} per posterior solve", star_time);
    println!(
        "node 5 posterior P(state=1): exact {:.6}, singleton-reweight {:.6}, node-star {:.6}",
        exact_posterior[5][1], singleton_posterior[5][1], star_posterior[5][1]
    );
    println!(
        "maximum posterior marginal error: singleton-reweight {:.6}, node-star {:.6}",
        max_error(&singleton_posterior),
        max_error(star_posterior)
    );
}
