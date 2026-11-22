use crate::ctbn::*;

use rand::distributions::{Distribution, Exp, Uniform};
use rand::{thread_rng, Rng};

pub struct Transition {
    location: usize,
    target: usize,
    tau: f64,
}

impl Transition {
    pub fn generate_transition(ctbn: &CTBN, state: &[usize]) -> Transition {
        Self::generate_transition_with_rng(ctbn, state, &mut thread_rng())
    }

    pub fn generate_transition_with_rng<R: Rng>(
        ctbn: &CTBN,
        state: &[usize],
        rng: &mut R,
    ) -> Transition {
        //draw location of transition & global survival time
        let mut cum_ext_rates: Vec<f64> = Vec::new();
        let mut cum_sum: f64 = 0.;
        for node in &ctbn.nodes {
            cum_sum += node.get_exit_rate(state.to_owned());
            cum_ext_rates.push(cum_sum);
        }

        let r_loc = Uniform::new(0., cum_sum);
        let loc = rng.sample(r_loc);

        //draw location of transition
        let location = cum_ext_rates.iter().position(|&x| x >= loc).unwrap();
        //draw surivial time
        let exp = Exp::new(cum_sum);

        let tau = exp.sample(rng);

        //draw transition given location
        let mut cum_rates: Vec<f64> = Vec::new();
        let mut cum_sum: f64 = 0.;
        let rates = &ctbn.nodes[location].get_transition_rates(state.to_owned());
        for rate in rates {
            cum_sum += rate;
            cum_rates.push(cum_sum);
        }
        let r_trans = Uniform::new(0., cum_sum);
        let trans = rng.sample(r_trans);

        let target = cum_rates.iter().position(|&x| x >= trans).unwrap();

        Transition {
            location,
            target,
            tau,
        }
    }
}

pub struct Sampler<'a> {
    ctbn: &'a CTBN,
    state: Vec<usize>,
    time: f64,
    time_max: f64,
    pub samples: Vec<(Vec<usize>, f64)>,
}

impl Sampler<'_> {
    pub fn create_sampler<'a>(ctbn: &'a CTBN, state: &[usize], time_max: &f64) -> Sampler<'a> {
        Sampler {
            ctbn,
            state: state.to_owned(),
            time: 0.,
            time_max: *time_max,
            samples: Vec::new(),
        }
    }

    fn propagate_with_rng<R: Rng>(&mut self, rng: &mut R) {
        let transition = Transition::generate_transition_with_rng(self.ctbn, &self.state, rng);
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

    fn update_state(&mut self, transition: Transition) {
        self.state[transition.location] = transition.target;
        self.time += transition.tau;
    }

    pub fn set_state(&mut self, state: &[usize]) {
        self.state = state.to_owned();
    }

    pub fn sample_path(&mut self) {
        self.sample_path_with_rng(&mut thread_rng());
    }

    pub fn sample_path_with_rng<R: Rng>(&mut self, rng: &mut R) {
        self.reset();
        while self.time <= self.time_max {
            self.propagate_with_rng(rng);
        }
    }

    pub fn state_at(&self, time: f64) -> Option<&[usize]> {
        if time < 0.0 {
            return None;
        }
        self.samples
            .iter()
            .take_while(|sample| sample.1 <= time)
            .last()
            .map(|sample| sample.0.as_slice())
    }
}
