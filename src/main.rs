use ndarray::prelude::*;
use ndarray::Array;
use rand::distributions::{IndependentSample,Gamma};
use rand::distributions::{Bernoulli, Distribution};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use rand::distributions::Exp;
use rand::seq::sample_slice;
use itertools::Itertools;
use mathru::special::gamma;
use mathru::special::gamma::ln_gamma;



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
    stats : STATS,
}

struct CTBN{
    nodes: Vec<NODE>
}


fn create_ctbn(adj: &Vec<Vec<usize>>, d: &Vec<usize>, params: &Vec<Vec<f64>>) -> CTBN{

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
    let stats: STATS =create_stats(d,p);
    NODE {
        index : index,
        d: d,
        params: params,
        parents: parents,
        parents_d : parents_d,
        cim : cim,
        stats: stats,
    }
}


fn create_cim(d: usize, p:  usize, alpha: f64, beta: f64) -> CIM {
    let gamma = Gamma::new(alpha, beta);
    let mut im = Array3::<f64>::zeros((d, d, p));

    for u in 0..p {
        for i in 0..d {
            for j in 0..d {
                im[[i, j, u]] = gamma.ind_sample(&mut rand::thread_rng());
            }
            im[[i,i,u]] = -im.slice(s![i,0..i,u]).sum() - im.slice(s![i,i+1..,u]).sum();
        }
    }
    CIM {
        d: d,
        p: p,
        val : im
    }
}

fn create_cim_glauber(d: usize, p:  usize, alpha: f64, beta: f64) -> CIM {
    let gamma = Gamma::new(alpha, beta);
    let mut im = Array3::<f64>::zeros((d, d, p));

    for u in 0..p {
        for i in 0..d {
            for j in 0..d {
                let rate = alpha*(1./2. + (-1. as f64 ).powf(i as f64)*((u as f64 )/(p as f64)))+alpha;
                im[[i, j, u]] = rate;
            }
            im[[i,i,u]] = -im.slice(s![i,0..i,u]).sum() - im.slice(s![i,i+1..,u]).sum();
        }
    }
    CIM {
        d: d,
        p: p,
        val : im
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
    let mut im =  &node.cim.val;
    let i = state[node.index];
    let u = get_condition(node,state);

    let mut a = im.slice(s![i,0..i,u]).clone();
    let mut b = im.slice(s![i,i+1..,u]).clone();

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
    samples: Vec<(Vec<usize>,f64)>,
}

fn create_sampler<'a>(ctbn: &'a CTBN, state: &Vec<usize>, time_max: &f64) -> SAMPLER<'a> {
    let mut samples:Vec<(Vec<usize>,f64)> =  Vec::new();
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
        self.samples.push(  (state,self.time) );
    }

    fn reset(&mut self) {
        let mut samples:Vec<(Vec<usize>,f64)> =  Vec::new();
        let state = self.state.clone();
        samples.push((state,0.));
        self.time =0.;
        self.samples = samples;

    }

    fn update_state(&mut self, transition: (usize, usize, f64)){
        self.state[transition.0]=transition.1;
        self.time = self.time + transition.2;
    }


    fn set_state(&mut self, state: &Vec<usize>){
        self.state = state.clone();
    }

    fn sample_path(&mut self){
        self.reset();
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

struct STATS{
    transitions: Array3::<u64>,
    survival_times: Array2::<f64>,
}

fn create_stats(d: usize, p: usize) -> STATS{
    let mut transitions = Array3::<u64>::zeros((d,d,p));
    let mut survival_times = Array2::<f64>::zeros((d,p));
    STATS{
        transitions: transitions,
        survival_times: survival_times,
    }
}

struct LEARNER  {
    d: Vec<usize>,
    params : Vec<Vec<f64>>,
    data : Vec<Vec<(Vec<usize>,f64)>>,
    ctbn: CTBN,
}

fn create_learner (adj: &Vec<Vec<usize>>, d: &Vec<usize>, params: &Vec<Vec<f64>> ) -> LEARNER {

    let mut ctbn = create_ctbn(&adj,&d,&params);
    LEARNER {
        d : d.clone(),
        params : params.clone(),
        data : Vec::new(),
        ctbn: ctbn,
    }
}

impl LEARNER {

    fn compute_stats(&mut self ) {
        for d in &self.data {

            let samples = d.clone();
            for i in 0..samples.len() - 1 {
                let s0 = samples[i].0.clone();
                let s1 = samples[i.clone() + 1].0.clone();
                let tau = samples[i.clone()+1].1.clone() - samples[i.clone()].1.clone();

                //find position where change happens between sample points
                let comp: Vec<bool> = s0.iter().zip(s1.iter()).map(|(&b, &v)| b != v).collect();
                let change: usize = comp.iter().find_position(|&&x| x == true).unwrap().0;

                let node = &mut self.ctbn.nodes[change.clone()];
                let u = get_condition(&node, s0.clone());

                let s  = s0.clone()[change.clone()];
                let s_ = s1.clone()[change.clone()];

                node.stats.transitions[[s,s_, u]] = node.stats.transitions[[s,s_, u.clone()]] + 1;
                node.stats.survival_times[[s, u.clone()]] = node.stats.survival_times[[s, u.clone()]] + tau;
            }
           // println!("{:?}",self.ctbn.nodes[0].stats.transitions);
            // println!("{:?}",self.ctbn.nodes[0].stats.survival_times);
        }

    }

    fn add_data(&mut self, samples : &Vec<(Vec<usize>,f64)> ) {
        self.data.push(samples.clone());
    }

    fn score_struct(&mut self, adj: &Vec<Vec<usize>>) -> (f64){
        let mut ctbn = create_ctbn(&adj,&self.d,&self.params);
        self.ctbn = ctbn;
        self.compute_stats();
        let mut score :f64 = 0.;

        for n in &self.ctbn.nodes {
            let m = n.stats.transitions.clone();
            let t = n.stats.survival_times.clone();
            for s in 0..n.d {
                for s_ in 0..n.d {
                    if (s != s_ ) {
                        for u in 0..n.parents_d.iter().product() {
                            score += ln_gamma(m[[s, s_, u]] as f64 + n.params[0]) - (m[[s, s_, u]] as f64 + n.params[0]-1.0) * (t[[s, u]] + n.params[1]).ln() - ln_gamma(n.params[0]) + (n.params[0]-1.0) * (n.params[1]).ln();
                        }
                    }
                }
            }
        }
        (score)
    }


    fn gen_all_adjs(&mut self, k: usize) -> (Vec<Vec<Vec<usize>>>){
        let mut par: Vec<usize> = Vec::new();
        let mut adjs: Vec<Vec<Vec<usize>>>= Vec::new();

        for i in 0..self.ctbn.nodes.len() {
            let mut pars:  Vec<Vec<usize>>= Vec::new();
            par = (0..self.ctbn.nodes.len()).collect();
            par = par.iter().filter(|&&x| x != i).cloned().collect_vec();
            for m in 0..k {
                pars.append(&mut par.iter().cloned().combinations(m).clone().collect_vec());
            }
            //pars.append(&mut par.iter().cloned().combinations(k).clone().collect_vec());
            adjs.push(pars.clone());
        }
        (adjs)
    }

    fn learn_structure (&mut self,k : usize) -> (f64, Vec<Vec<usize>>,Vec<f64>) {

        let mut scores:Vec<f64> = Vec::new();
        let adjs = self.gen_all_adjs(k);
        let combs = (0..self.ctbn.nodes.len()).map(|x| (0..adjs[x].len())).multi_cartesian_product().collect_vec(); // gen cartension prodcut over all adjs indices (all structures)
        //println!("{:?}",combs);
        let mut max_score = -f64::INFINITY;
        let mut max_adj : Vec<Vec<usize>> = Vec::new();
        for z in combs {
            let mut adj : Vec<Vec<usize>> = Vec::new();
            for i in 0..z.len() {
                adj.push(  adjs[i][z[i]].clone());
            }

            let score = self.score_struct(&adj) ;
            if (score > max_score) {
                max_score = score.clone();
                max_adj   = adj.clone();
            }
            scores.push(score);
        }
        //let max_score = scores.iter().cloned().fold(0./0., f64::max);
        (max_score,max_adj,scores)
    }

    fn expected_structure(&mut self, scores: Vec<f64>, k : usize) -> (Array2::<f64>) {
        let adjs = self.gen_all_adjs(k);
        let mut w = scores.clone();
        let norm = scores.iter().sum::<f64>() as f64;
        for k in 0..scores.len() {
            w[k] = scores.clone()[k] / norm;
        }
        let combs = (0..self.ctbn.nodes.len()).map(|x| (0..adjs[x].len())).multi_cartesian_product().collect_vec(); // gen cartension prodcut over all adjs indices (all structures)
        //println!("{:?}",combs);
        let mut exp_struct = Array2::<f64>::zeros((self.ctbn.nodes.len(),self.ctbn.nodes.len()));

        let mut k = 0;
        for z in combs {
            for i in 0..z.len() {
                for j in adjs[i][z[i]].clone() {
                    exp_struct[[i,j]] +=w[k];
                }
            }
            k = k + 1 ;
        }
        (exp_struct)
    }
}

fn main() {

    let adj: Vec<Vec<usize>> =vec![vec![1,2],vec![0],vec![]];
    let d: Vec<usize> = vec![3,3,3];
    let params:Vec<Vec<f64>> = vec![vec![1.,4.],vec![1.,4.],vec![1.,4.]];

    let ctbn = create_ctbn(&adj,&d,&params);
    let mut state: Vec<usize> = vec![1,1,1];
    let mut sampler: SAMPLER = create_sampler(&ctbn, &state,&10.);

    let params:Vec<Vec<f64>> = vec![vec![1.,1.],vec![1.,1.],vec![1.,1.]];
    let mut learner: LEARNER = create_learner(&adj,&d,&params);

    let d = Bernoulli::new(0.5);
    for i in 0..50 {
        for j in 0..3 {
            let v = d.sample(&mut rand::thread_rng()) as usize;
            state[j] = v;
        }
        //sampler.reset();
        sampler.set_state(&state);
        sampler.sample_path();
       // println!("{:?}",sampler.samples);
        learner.add_data(&sampler.samples);

    }

    let adj0: Vec<Vec<usize>> =vec![vec![],vec![2],vec![1]];
    let score = learner.score_struct(&adj);
    println!("{:?}",score);
    let score = learner.score_struct(&adj0);
    println!("{:?}",score);
   // learner.score_struct(&adj);
    let out = learner.learn_structure(3);
    println!("{:?}",out);
    //TODO:
    // create crate for ctbns - sampler
    // learn from paths


}