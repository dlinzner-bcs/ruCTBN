use ndarray::prelude::*;
use ndarray::Array;

pub fn convert2dec(input: Vec<usize>, base: &Vec<usize>) -> usize {
    let mut dec: usize = 0;
    let mut k: usize = 0;
    for c in input {
        let mut a = 1;
        for b in 0..k {
            a = a * base[b];
        }
        dec = dec + a * c;
        k = k + 1;
    }
    dec
}

pub struct Stats {
    pub transitions: Array3<f64>,
    pub survival_times: Array2<f64>,
}

impl Stats {
    pub fn create_stats(d: usize, p: usize) -> Stats {
        let mut transitions = Array3::<f64>::zeros((d, d, p));
        let mut survival_times = Array2::<f64>::zeros((d, p));
        Stats {
            transitions: transitions,
            survival_times: survival_times,
        }
    }
}
