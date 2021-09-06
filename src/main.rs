mod common;
mod ctbn;
mod learner;
mod sampler;

use crate::sampler::Sampler;
use ctbn::CTBN;
use learner::Learner;
use rand::distributions::{Bernoulli, Distribution};

fn main() {
    let adj: Vec<Vec<usize>> = vec![vec![], vec![1, 2], vec![]];
    let d: Vec<usize> = vec![3, 3, 3];
    let params: Vec<Vec<f64>> = vec![vec![10., 4.], vec![1., 4.], vec![1., 4.]];

    let ctbn = CTBN::create_ctbn(&adj, &d, &params);
    let mut state: Vec<usize> = vec![1, 1, 1];
    let mut sampler: Sampler = Sampler::create_sampler(&ctbn, &state, &1.);

    let params: Vec<Vec<f64>> = vec![vec![1., 1.], vec![1., 1.], vec![1., 1.]];
    let mut learner: Learner = Learner::create_learner(&adj, &d, &params);

    let d = Bernoulli::new(0.5);
    for _ in 0..1000 {
        for j in 0..3 {
            let v = d.sample(&mut rand::thread_rng()) as usize;
            state[j] = v;
        }
        //sampler.reset();
        sampler.set_state(&state);
        sampler.sample_path();
        // println!("{:?}",sampler.samples);
        //println!("{:?}",sampler.samples);
        learner.add_data(&sampler.samples);
    }
    // learner.score_struct(&adj);
    let out = learner.learn_structure(3);
    println!("{:?}", out);
    let out = learner.learn_structure(3);
    println!("{:?}", out);
    //TODO:
    // create crate for ctbns - sampler
    // learn from path
}
