use crate::common::{convert2dec, Stats};
use ndarray::prelude::*;
use rand::distributions::{Distribution, Gamma};

#[derive(Debug)]
struct CIM {
    pub d: usize,
    p: usize,
    val: Array3<f64>,
}

impl CIM {
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
        CIM {
            d: d,
            p: p,
            val: im,
        }
    }

    #[allow(dead_code)]
    pub fn create_cim_glauber(d: usize, p: usize, alpha: f64) -> CIM {
        let mut im = Array3::<f64>::zeros((d, d, p));

        for u in 0..p {
            for i in 0..d {
                for j in 0..d {
                    let rate = alpha
                        * (1. / 2. + (-1. as f64).powf(i as f64) * ((u as f64) / (p as f64)))
                        + alpha;
                    im[[i, j, u]] = rate;
                }
                im[[i, i, u]] = -im.slice(s![i, 0..i, u]).sum() - im.slice(s![i, i + 1.., u]).sum();
            }
        }
        CIM {
            d: d,
            p: p,
            val: im,
        }
    }
}

pub struct NODE {
    index: usize,
    pub d: usize,
    pub params: Vec<f64>,
    parents: Vec<usize>,
    pub parents_d: Vec<usize>,
    cim: CIM,
    pub stats: Stats,
}

impl NODE {
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

        let a = im.slice(s![i, 0..i, u]).clone();
        let b = im.slice(s![i, i + 1.., u]).clone();

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
    pub fn create_ctbn(adj: &Vec<Vec<usize>>, d: &Vec<usize>, params: &Vec<Vec<f64>>) -> CTBN {
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

        CTBN { nodes: nodes }
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
            index: index,
            d: d,
            params: params,
            parents: parents,
            parents_d: parents_d,
            cim: cim,
            stats: stats,
        }
    }
}

pub fn get_condition(node: &NODE, state: Vec<usize>) -> usize {
    let mut par_state: Vec<usize> = Vec::new();

    for p in &node.parents {
        par_state.push(state[p.clone()]);
    }
    convert2dec(par_state, &node.parents_d)
}
