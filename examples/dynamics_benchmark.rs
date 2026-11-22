use ctbn::ctbn::CTBN;
use ctbn::dynamics::{cluster_variational_dynamics, exact_dynamics};
use ctbn::sampler::Sampler;
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
}
