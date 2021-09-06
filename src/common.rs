use ndarray::prelude::*;

pub fn convert2dec(input: &[usize], base: &[usize]) -> usize {
    let mut dec: usize = 0;
    let mut k: usize = 0;
    for c in input {
        let mut a = 1;
        for b in 0..k {
            a *= base[b];
        }
        dec += a * c;
        k += 1;
    }
    dec
}

pub struct Stats {
    pub transitions: Array3<f64>,
    pub survival_times: Array2<f64>,
}

impl Stats {
    pub fn create_stats(d: usize, p: usize) -> Stats {
        let transitions = Array3::<f64>::zeros((d, d, p));
        let survival_times = Array2::<f64>::zeros((d, p));
        Stats {
            transitions,
            survival_times,
        }
    }
}
