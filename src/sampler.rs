use crate::ctbn::*;
use rand::distributions::{Exp, Uniform};
use rand::distributions::{Gamma, IndependentSample};
use rand::{thread_rng, Rng};

pub struct Sampler<'a> {
    ctbn: &'a CTBN,
    state: Vec<usize>,
    time: f64,
    time_max: f64,
    pub samples: Vec<(Vec<usize>, f64)>,
}

impl Sampler<'_> {
    pub fn create_sampler<'a>(ctbn: &'a CTBN, state: &Vec<usize>, time_max: &f64) -> Sampler<'a> {
        let mut samples: Vec<(Vec<usize>, f64)> = Vec::new();
        Sampler {
            ctbn: ctbn,
            state: state.clone(),
            time: 0.,
            time_max: time_max.clone(),
            samples: samples,
        }
    }

    fn propagate(&mut self) {
        let transition = generate_transition(&self.ctbn, &self.state);
        self.update_state(transition);
        let state = self.state.clone();
        self.samples.push((state, self.time));
    }

    fn reset(&mut self) {
        let mut samples: Vec<(Vec<usize>, f64)> = Vec::new();
        let state = self.state.clone();
        samples.push((state, 0.));
        self.time = 0.;
        self.samples = samples;
    }

    fn update_state(&mut self, transition: (usize, usize, f64)) {
        self.state[transition.0] = transition.1;
        self.time = self.time + transition.2;
    }

    pub fn set_state(&mut self, state: &Vec<usize>) {
        self.state = state.clone();
    }

    pub fn sample_path(&mut self) {
        self.reset();
        while self.time <= self.time_max {
            self.propagate();
        }
    }
}

fn generate_transition(ctbn: &CTBN, state: &Vec<usize>) -> (usize, usize, f64) {
    //draw location of transition & global survival time
    let mut cum_ext_rates: Vec<f64> = Vec::new();
    let mut cum_sum: f64 = 0.;
    for node in &ctbn.nodes {
        cum_sum = cum_sum + get_exit_rate(node, state.clone());
        cum_ext_rates.push(cum_sum);
    }

    let mut rng = thread_rng();
    let r_loc = Uniform::new(0., cum_sum);
    let loc = rng.sample(r_loc);

    //draw location of transition
    let location = cum_ext_rates.iter().position(|&x| x >= loc).unwrap();
    //draw surivial time
    let exp = Exp::new(cum_sum);
    let tau = exp.ind_sample(&mut rand::thread_rng());

    //draw transition given location
    let mut cum_rates: Vec<f64> = Vec::new();
    let mut cum_sum: f64 = 0.;
    let rates = get_transition_rates(&ctbn.nodes[location], state.clone());
    for rate in rates {
        cum_sum = cum_sum + rate;
        cum_rates.push(cum_sum);
    }
    let mut rng = thread_rng();
    let r_trans = Uniform::new(0., cum_sum);
    let trans = rng.sample(r_trans);

    let transition = cum_rates.iter().position(|&x| x >= trans).unwrap();

    (location, transition, tau)
}
