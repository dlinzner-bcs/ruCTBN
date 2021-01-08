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


fn create_stats(d: usize, p: usize) -> STATS{
    let mut transitions = Array3::<f64>::zeros((d,d,p));
    let mut survival_times = Array2::<f64>::zeros((d,p));
    STATS{
        transitions: transitions,
        survival_times: survival_times,
    }
}

fn create_ctbn(adj: &[Vec<usize>], d: &[usize], params: &[Vec<f64>]) -> CTBN{

    assert_eq!(adj.len(),d.len());
    assert_eq!(d.len(),params.len());

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
        cim : cim,
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

struct SAMPLER <'a>{
    ctbn: &'a CTBN,
    state: Vec<usize>,
    time: f64,
    time_max: f64,
    samples: Vec<Vec<usize>>,
}

fn create_sampler<'a>(ctbn: &'a CTBN, state: &Vec<usize>, time_max: &f64) -> SAMPLER<'a> {
    let mut samples:Vec<Vec<(usize)>> =  Vec::new();

    SAMPLER{
        ctbn: ctbn,
        state : state.clone(),
        time : 0.,
        time_max : time_max.clone(),
        samples :  samples,
    }

}

impl SAMPLER <'_>{

    fn propagate(&mut self) {
        let transition = generate_transition(&self.ctbn, &self.state);
        self.update_state(transition);
        let state = self.state.clone();
        self.samples.push(  state);
    }

    fn update_state(&mut self, transition: (usize, usize, f64)){
        self.state[transition.0]=transition.1;
        self.time = self.time + transition.2;
    }

     fn update_stat(&mut self, transition: (usize, usize, usize, f64)){

    }

    fn set_state(&mut self, state: &Vec<usize>){
        self.state = state.clone();
    }

    fn sample_path(&mut self){

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

    (location,transition,tau)

}

struct LEARNER <'a> {
    ctbn:&'a CTBN,
    stats: stats,
}

struct STATS{
    transitions: Array3::<f64>,
    survival_times: Array2::<f64>,
}

fn main() {

    let adj: [Vec<usize>;6] = [vec![1],vec![2],vec![1,2],vec![4],vec![0,5],vec![1]];
    let d: [usize;6] = [2,2,2,2,2,2];
    let params:[Vec<f64>;6] = [vec![1.,1.],vec![1.,1.],vec![1.,1.],vec![1.,1.],vec![1.,1.],vec![1.,1.]];

    let ctbn = create_ctbn(&adj,&d,&params);
    let state: Vec<usize> = vec![1,1,1,1,1,1,1];
    let mut sampler: SAMPLER = create_sampler(&ctbn, &state,&1.);

    sampler.sample_path();
    println!("{:?}", sampler.samples);


    //TODO:
    // create crate for ctbns - sampler
    // learn from paths


}