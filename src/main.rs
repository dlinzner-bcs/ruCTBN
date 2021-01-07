use ndarray::prelude::*;
use ndarray::Array;
use rand::distributions::{IndependentSample,Gamma};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use rand::distributions::Exp;

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

fn get_exit_rate(node: &NODE, state: Vec<usize>) -> (f64){
    let c =  &node.cim;
    (- c.val[[state[node.index], state[node.index],get_condition(node,state)]])
}

fn get_transition_rates(node: &NODE, state: Vec<usize>) -> (Vec<f64>){
    let mut IM =  &node.cim.val;
    let i = state[node.index];
    let u = get_condition(node,state);

    let mut a = IM.slice(s![i,0..i,u]).clone();
    let mut b = IM.slice(s![i,i+1..,u]).clone();

    let mut out = a.to_vec() ;
    out.push(0.);
    out.append(&mut b.to_vec());
    (out)


}


fn generate_transition(ctbn: &CTBN, state: Vec<usize>) -> (usize,usize,f64){

    //draw location of transition & global survival time
        let mut cum_ext_rates: Vec<f64> = Vec::new();
        let mut cum_sum: f64 = 0.;
        for node in &ctbn.nodes {
            cum_sum = cum_sum + get_exit_rate(node,state.clone());
            cum_ext_rates.push(cum_sum );
        }

        let mut rng = thread_rng();
        let r_loc = Uniform::new(0., cum_sum);
        let loc = rng.sample(r_loc);

        //draw location of transition
        let mut location = cum_ext_rates.iter().position(|&x| x>=loc).unwrap();
        //draw surivial time
        let exp = Exp::new(cum_sum);
        let tau = exp.ind_sample(&mut rand::thread_rng());

    //draw transition given location
        let mut cum_rates: Vec<f64> = Vec::new();
        let mut cum_sum: f64 = 0.;
        let rates = get_transition_rates(&ctbn.nodes[location],state.clone());
        for rate in rates {
            cum_sum = cum_sum + rate;
            cum_rates.push(cum_sum );
        }
        let mut rng = thread_rng();
        let r_trans = Uniform::new(0., cum_sum);
        let trans = rng.sample(r_trans);

        let mut transition = cum_rates.iter().position(|&x| x>=trans).unwrap();


    (location,transition,tau)
}


fn main() {

    let adj: [Vec<usize>;3] = [vec![1],vec![2],vec![1,2]];
    let d: [usize;3] = [4,4,4];
    let params:[Vec<f64>;3] = [vec![1.,1.],vec![1.,1.],vec![1.,1.]];

    let ctbn = create_ctbn(&adj,&d,&params);

    let state: Vec<usize> = vec![2,1,2];

    let node = &ctbn.nodes[2]; // for the noob: copy needs to be implemented for NODE struct - or just reference the original object - needs to be considered if NODE is referenced often
    println!("{:?}", generate_transition(&ctbn,state));


}