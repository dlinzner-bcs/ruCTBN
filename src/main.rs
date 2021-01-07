use ndarray::prelude::*;
use ndarray::Array;
use rand::distributions::{IndependentSample,Gamma};


#[derive(Debug)]
struct CIM {
    d:  usize,
    p:  usize,
    val : Array3::<f64>
}

struct NODE {
    index: usize,
    d: usize,
    params: Vec<f64>,
    parents: Vec<usize>,
    parents_d: Vec<usize>,
    cim: CIM,
}

struct CTBN{
    nodes: Vec<NODE>
}

struct SAMPLE{
    states: Vec<usize>,
    times:  Vec<f64>,
}

struct SAMPLER{
    ctbn: CTBN,
    inital_state: Vec<usize>,
    init_time: f64,
    samples: Vec<SAMPLE>,
}


fn create_ctbn(adj: &[Vec<usize>], d: &[usize], params: &[Vec<f64>]) -> CTBN{
    let mut nodes = Vec::new();

    for i in 0..adj.len(){
        let parents = &adj[i];

        let mut parents_d : Vec<usize> = Vec::new();
        for k in parents.clone(){
            parents_d.push( d[k]);
        }

        let param = &params[i];
        let node: NODE = create_node(i, d[i], param.clone(), parents.clone(), parents_d);

        nodes.push(node);
    }

    CTBN{
        nodes: nodes,
    }
}

fn create_node(index: usize, d: usize, params: Vec<f64>, parents: Vec<usize>, parents_d: Vec<usize>) -> NODE {
    let p : usize = parents_d.iter().product() ;
    let cim = create_cim(d,p,params[0], params[1]);

    NODE {
        index : index,
        d: d,
        params: params,
        parents: parents,
        parents_d : parents_d,
        cim : cim
    }
}

fn create_cim(d: usize, p:  usize, alpha: f64, beta: f64) -> CIM {
    let gamma = Gamma::new(alpha, beta);
    let mut IM = Array3::<f64>::zeros((d,d,p));

    for u in 0..p {
        for i in 0..d {
            for j in 0..d {
                IM[[i, j, u]] = gamma.ind_sample(&mut rand::thread_rng());
            }
            IM[[i,i,u]] = -IM.slice(s![i,0..i,u]).sum() -IM.slice(s![i,i+1..,u]).sum();
        }
    }
    CIM {
        d: d,
        p: p,
        val : IM
    }
}

fn  convert2dec(input: Vec<usize>,  base: &Vec<usize>) -> usize {
    let mut dec: usize = 0;
    let mut k: usize = 0;
    for c in input {
        let mut a = 1;
        for b in 0..k {
            a = a*base[b];
        }
        dec = dec + a*c;
        k = k + 1;
    }
    dec
}

fn get_condition(node: &NODE, state: Vec<usize>) -> usize {
    let mut par_state : Vec<usize> = Vec::new();

    for p in &node.parents{
        par_state.push(state[p.clone()]);
    }
    convert2dec(par_state,&node.parents_d)
}

fn get_exit_rate(sampler: SAMPLER) -> f64{
    1.0
}

fn generate_sample(sampler: SAMPLER) -> SAMPLE{
    let state : Vec<usize> = sampler.inital_state;
    for node in sampler.ctbn.nodes {

    }
    SAMPLE{
        states : vec![1],
        times  : vec![1.],
    }
}


fn main() {

    let d = 2;
    let p = 1;

    let adj: [Vec<usize>;3] = [vec![1],vec![2],vec![1,2]];
    let d: [usize;3] = [3,2,3];
    let params:[Vec<f64>;3] = [vec![0.1,0.1],vec![0.1,0.1],vec![0.1,0.1]];

    let ctbn = create_ctbn(&adj,&d,&params);

    let state: Vec<usize> = vec![2,1,2];
    let base: Vec<usize> = vec![3,2,3];

    let node = &ctbn.nodes[2];
    println!("{:?}", get_condition(&node,state));


}