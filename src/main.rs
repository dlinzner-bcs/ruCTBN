mod common;
mod ctbn;
mod learner;
mod sampler;

use crate::sampler::Sampler;
use ctbn::CTBN;
use learner::Learner;
use rand::distributions::{Bernoulli, Distribution};

fn main() {
    let adj: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![]];
    let d: Vec<usize> = vec![2, 2, 2];
    let params: Vec<Vec<f64>> = vec![vec![1., 2.], vec![1., 2.], vec![1., 2.]];

    let ctbn = CTBN::create_ctbn(&adj, &d, &params);
    let mut state: Vec<usize> = vec![1, 1, 1];
    let mut sampler: Sampler = Sampler::create_sampler(&ctbn, &state, &1.);

    let params: Vec<Vec<f64>> = vec![vec![1., 1.], vec![1., 1.], vec![1., 1.]];
    let mut learner: Learner = Learner::create_learner(&adj, &d, &params);

    let d = Bernoulli::new(0.5);
    for _ in 0..100 {
        for j in 0..3 {
            let v = d.sample(&mut rand::thread_rng()) as usize;
            state[j] = v;
        }
        sampler.set_state(&state);
        sampler.sample_path();
        learner.add_data(&sampler.samples);
    }
    let out = learner.learn_structure(3);
    println!("{:?}", out);
}
