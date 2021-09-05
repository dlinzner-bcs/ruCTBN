use crate::ctbn::*;
use itertools::Itertools;
use ndarray::prelude::*;
use statrs::function::gamma::ln_gamma;

pub struct Learner {
    d: Vec<usize>,
    params: Vec<Vec<f64>>,
    data: Vec<Vec<(Vec<usize>, f64)>>,
    ctbn: CTBN,
}

impl Learner {
    pub fn create_learner(
        adj: &Vec<Vec<usize>>,
        d: &Vec<usize>,
        params: &Vec<Vec<f64>>,
    ) -> Learner {
        let ctbn = CTBN::create_ctbn(&adj, &d, &params);
        Learner {
            d: d.clone(),
            params: params.clone(),
            data: Vec::new(),
            ctbn: ctbn,
        }
    }

    fn compute_stats(&mut self) {
        for d in &self.data {
            let samples = d.clone();
            for i in 0..(samples.len() - 1) {
                let s0 = samples[i].0.clone();
                let s1 = samples[i.clone() + 1].0.clone();
                let tau = samples[i.clone() + 1].1.clone() - samples[i.clone()].1.clone();

                //find position where change happens between sample points
                let comp: Vec<bool> = s0.iter().zip(s1.iter()).map(|(&b, &v)| b != v).collect();
                let change: usize = comp.iter().find_position(|&&x| x == true).unwrap().0;

                let node = &mut self.ctbn.nodes[change.clone()];
                let u = get_condition(&node, s0.clone());

                let s = s0.clone()[change.clone()];
                let s_ = s1.clone()[change.clone()];

                node.stats.transitions[[s, s_, u]] =
                    node.stats.transitions[[s, s_, u.clone()]] + 1.;
                node.stats.survival_times[[s, u.clone()]] =
                    node.stats.survival_times[[s, u.clone()]] + tau;
            }
        }
    }

    pub fn add_data(&mut self, samples: &Vec<(Vec<usize>, f64)>) {
        self.data.push(samples.clone());
    }

    fn score_struct(&mut self, adj: &Vec<Vec<usize>>) -> f64 {
        let ctbn = CTBN::create_ctbn(&adj, &self.d, &self.params);
        self.ctbn = ctbn;
        self.compute_stats();
        let mut score: f64 = 0.;

        for n in &self.ctbn.nodes {
            let m = n.stats.transitions.clone();
            let t = n.stats.survival_times.clone();
            for s in 0..n.d {
                for s_ in 0..n.d {
                    if s != s_ {
                        for u in 0..n.parents_d.iter().product() {
                            score += ln_gamma(m[[s, s_, u]] + n.params[0])
                                - (m[[s, s_, u]] + n.params[0]) * (t[[s, u]] + n.params[1]).ln()
                                - ln_gamma(n.params[0])
                                + (n.params[0]) * (n.params[1]).ln();
                        }
                    }
                }
            }
        }
        score
    }

    fn gen_all_adjs(&mut self, k: usize) -> Vec<Vec<Vec<usize>>> {
        let mut par: Vec<usize>;
        let mut adjs: Vec<Vec<Vec<usize>>> = Vec::new();

        for i in 0..self.ctbn.nodes.len() {
            let mut pars: Vec<Vec<usize>> = Vec::new();
            par = (0..self.ctbn.nodes.len()).collect();
            par = par.iter().filter(|&&x| x != i).cloned().collect_vec();
            for m in 0..k {
                pars.append(&mut par.iter().cloned().combinations(m).clone().collect_vec());
            }
            //pars.append(&mut par.iter().cloned().combinations(k).clone().collect_vec());
            adjs.push(pars.clone());
        }
        adjs
    }

    pub fn learn_structure(&mut self, k: usize) -> (f64, Vec<Vec<usize>>, Vec<f64>) {
        let mut scores: Vec<f64> = Vec::new();
        let adjs = self.gen_all_adjs(k);
        let combs = (0..self.ctbn.nodes.len())
            .map(|x| (0..adjs[x].len()))
            .multi_cartesian_product()
            .collect_vec(); // gen cartension prodcut over all adjs indices (all structures)
                            //println!("{:?}",combs);
        let mut max_score = -f64::INFINITY;
        let mut max_adj: Vec<Vec<usize>> = Vec::new();
        for z in combs {
            let mut adj: Vec<Vec<usize>> = Vec::new();
            for i in 0..z.len() {
                adj.push(adjs[i][z[i]].clone());
            }

            let score = self.score_struct(&adj);
            if score > max_score {
                max_score = score.clone();
                max_adj = adj.clone();
            }
            scores.push(score);
        }
        //let max_score = scores.iter().cloned().fold(0./0., f64::max);
        (max_score, max_adj, scores)
    }

    #[allow(dead_code)]
    fn expected_structure(&mut self, scores: Vec<f64>, k: usize) -> Array2<f64> {
        let adjs = self.gen_all_adjs(k);
        let mut w = scores.clone();
        let norm = scores.iter().sum::<f64>() as f64;
        for k in 0..scores.len() {
            w[k] = scores.clone()[k] / norm;
        }
        let combs = (0..self.ctbn.nodes.len())
            .map(|x| (0..adjs[x].len()))
            .multi_cartesian_product()
            .collect_vec(); // gen cartension prodcut over all adjs indices (all structures)
                            //println!("{:?}",combs);
        let mut exp_struct = Array2::<f64>::zeros((self.ctbn.nodes.len(), self.ctbn.nodes.len()));

        let mut k = 0;
        for z in combs {
            for i in 0..z.len() {
                for j in adjs[i][z[i]].clone() {
                    exp_struct[[i, j]] += w[k];
                }
            }
            k = k + 1;
        }
        exp_struct
    }
}
