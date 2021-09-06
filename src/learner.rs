use crate::ctbn::*;
use itertools::Itertools;
use ndarray::prelude::*;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

pub struct Learner {
    d: Vec<usize>,
    params: Vec<Vec<f64>>,
    data: Vec<Vec<(Vec<usize>, f64)>>,
    ctbn: CTBN,
}

impl Learner {
    pub fn create_learner(adj: &[Vec<usize>], d: &[usize], params: &[Vec<f64>]) -> Learner {
        let ctbn = CTBN::create_ctbn(&adj, d, params);
        Learner {
            d: d.to_owned(),
            params: params.to_owned(),
            data: Vec::new(),
            ctbn,
        }
    }

    fn compute_stats(&mut self) {
        for d in &self.data {
            let samples = d.clone();
            for i in 0..(samples.len() - 1) {
                let s0 = samples[i].0.clone();
                let s1 = samples[i + 1].0.clone();
                let tau = samples[i + 1].1 - samples[i].1;

                //find position where change happens between sample points
                let comp: Vec<bool> = s0.iter().zip(s1.iter()).map(|(&b, &v)| b != v).collect();
                let change: usize = comp.iter().find_position(|&&x| x).unwrap().0;

                let node = &mut self.ctbn.nodes[change];
                let u = get_condition(node, s0.clone());

                let s = s0.clone()[change];
                let s_ = s1.clone()[change];

                node.stats.transitions[[s, s_, u]] = node.stats.transitions[[s, s_, u]] + 1.;
                node.stats.survival_times[[s, u]] = node.stats.survival_times[[s, u]] + tau;
            }
        }
    }

    pub fn add_data(&mut self, samples: &[(Vec<usize>, f64)]) {
        self.data.push(samples.to_owned());
    }

    fn score_struct(&mut self, adj: &[Vec<usize>]) -> f64 {
        let ctbn = CTBN::create_ctbn(adj, &self.d, &self.params);
        self.ctbn = ctbn;
        self.compute_stats();

        self.ctbn
            .nodes
            .par_iter()
            .map(|n| {
                let m = n.stats.transitions.clone();
                let t = n.stats.survival_times.clone();
                (0..n.d)
                    .cartesian_product(0..n.d)
                    .filter(|&(s, s_)| s != s_)
                    .map(|(s, s_)| {
                        (0..n.parents_d.iter().product())
                            .map(|u| {
                                ln_gamma(m[[s, s_, u]] + n.params[0])
                                    - (m[[s, s_, u]] + n.params[0]) * (t[[s, u]] + n.params[1]).ln()
                                    - ln_gamma(n.params[0])
                                    + (n.params[0]) * (n.params[1]).ln()
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
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
                max_score = score;
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
        let norm = scores.iter().sum::<f64>() as f64;
        let weight = scores.iter().map(|w| w / norm).collect_vec();

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
                    exp_struct[[i, j]] += weight[k];
                }
            }
            k += 1;
        }
        exp_struct
    }
}
