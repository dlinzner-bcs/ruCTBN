use crate::common::{convert2dec, Stats};
use ndarray::prelude::*;
use rand::distributions::{Distribution, Gamma};

#[derive(Debug)]
pub struct CIM {
    pub d: usize,
    pub p: usize,
    pub val: Array3<f64>,
}

impl CIM {
    pub fn from_rates(d: usize, p: usize, val: Array3<f64>) -> CIM {
        assert_eq!(val.dim(), (d, d, p));
        for condition in 0..p {
            for from in 0..d {
                for to in 0..d {
                    assert!(
                        val[[from, to, condition]].is_finite(),
                        "transition rates must be finite"
                    );
                    if from != to {
                        assert!(
                            val[[from, to, condition]] >= 0.0,
                            "off-diagonal transition rates must be non-negative"
                        );
                    }
                }
                let exit_rate: f64 = (0..d)
                    .filter(|&to| to != from)
                    .map(|to| val[[from, to, condition]])
                    .sum();
                assert!(
                    (val[[from, from, condition]] + exit_rate).abs() < 1e-10,
                    "diagonal entries must be the negative sum of the off-diagonal rates"
                );
            }
        }
        CIM { d, p, val }
    }

    pub fn create_cim(d: usize, p: usize, alpha: f64, beta: f64) -> CIM {
        let gamma = Gamma::new(alpha, beta);
        let mut im = Array3::<f64>::zeros((d, d, p));

        for u in 0..p {
            for i in 0..d {
                for j in 0..d {
                    im[[i, j, u]] = gamma.sample(&mut rand::thread_rng());
                }
                im[[i, i, u]] = -im.slice(s![i, 0..i, u]).sum() - im.slice(s![i, i + 1.., u]).sum();
            }
        }
        CIM { d, p, val: im }
    }

    #[allow(dead_code)]
    pub fn create_cim_glauber(d: usize, p: usize, alpha: f64) -> CIM {
        let mut im = Array3::<f64>::zeros((d, d, p));

        for u in 0..p {
            for i in 0..d {
                for j in 0..d {
                    let rate = alpha
                        * (1. / 2. + (-1_f64).powf(i as f64) * ((u as f64) / (p as f64)))
                        + alpha;
                    im[[i, j, u]] = rate;
                }
                im[[i, i, u]] = -im.slice(s![i, 0..i, u]).sum() - im.slice(s![i, i + 1.., u]).sum();
            }
        }
        CIM { d, p, val: im }
    }
}

pub struct NODE {
    index: usize,
    pub d: usize,
    pub params: Vec<f64>,
    parents: Vec<usize>,
    pub parents_d: Vec<usize>,
    pub cim: CIM,
    pub stats: Stats,
}

impl NODE {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn parents(&self) -> &[usize] {
        &self.parents
    }

    pub fn transition_rate(&self, from: usize, to: usize, condition: usize) -> f64 {
        assert!(from < self.d && to < self.d && condition < self.cim.p);
        self.cim.val[[from, to, condition]]
    }

    pub fn get_exit_rate(&self, state: Vec<usize>) -> f64 {
        let c = &self.cim;
        -c.val[[
            state[self.index],
            state[self.index],
            get_condition(self, state),
        ]]
    }

    pub fn get_transition_rates(&self, state: Vec<usize>) -> Vec<f64> {
        let im = &self.cim.val;
        let i = state[self.index];
        let u = get_condition(self, state);

        let a = im.slice(s![i, 0..i, u]);
        let b = im.slice(s![i, i + 1.., u]);

        let mut out = a.to_vec();
        out.push(0.);
        out.append(&mut b.to_vec());
        out
    }
}

pub struct CTBN {
    pub nodes: Vec<NODE>,
}

impl CTBN {
    pub fn create_ctbn_from_rates(
        adj: &[Vec<usize>],
        d: &[usize],
        rate_matrices: &[Array3<f64>],
    ) -> CTBN {
        assert_eq!(adj.len(), d.len());
        assert_eq!(d.len(), rate_matrices.len());

        let mut nodes = Vec::new();
        for (index, parents) in adj.iter().enumerate() {
            for &parent in parents {
                assert!(parent < d.len(), "parent index is outside the CTBN");
            }
            let parents_d: Vec<usize> = parents.iter().map(|&parent| d[parent]).collect();
            let p = parents_d.iter().product();
            let cim = CIM::from_rates(d[index], p, rate_matrices[index].clone());
            nodes.push(NODE {
                index,
                d: d[index],
                params: Vec::new(),
                parents: parents.clone(),
                parents_d,
                cim,
                stats: Stats::create_stats(d[index], p),
            });
        }
        CTBN { nodes }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        self.nodes.iter().map(|node| node.d).collect()
    }

    pub fn create_ctbn(adj: &[Vec<usize>], d: &[usize], params: &[Vec<f64>]) -> CTBN {
        assert_eq!(adj.len(), d.len());
        assert_eq!(d.len(), params.len());

        let mut nodes = Vec::new();

        for i in 0..adj.len() {
            let parents = &adj[i];

            let mut parents_d: Vec<usize> = Vec::new();
            for k in parents.clone() {
                parents_d.push(d[k]);
            }

            let param = &params[i];
            let node: NODE = CTBN::create_node(i, d[i], param.clone(), parents.clone(), parents_d);

            nodes.push(node);
        }

        CTBN { nodes }
    }

    fn create_node(
        index: usize,
        d: usize,
        params: Vec<f64>,
        parents: Vec<usize>,
        parents_d: Vec<usize>,
    ) -> NODE {
        let p: usize = parents_d.iter().product();
        let cim = CIM::create_cim(d, p, params[0], params[1]);
        let stats: Stats = Stats::create_stats(d, p);
        NODE {
            index,
            d,
            params,
            parents,
            parents_d,
            cim,
            stats,
        }
    }
}

pub fn get_condition(node: &NODE, state: Vec<usize>) -> usize {
    let mut par_state: Vec<usize> = Vec::new();

    for p in &node.parents {
        par_state.push(state[*p]);
    }
    convert2dec(&par_state, &node.parents_d)
}
