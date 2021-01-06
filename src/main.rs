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
    params: Array2::<f64>,
    parents: Vec<u32>,
    parents_d: Vec<usize>,
    cim: CIM,
}

struct CTBN{
    nodes: Vec<NODE>
}

fn create_ctbn(adj: &[Vec<usize>], d: &[usize], params: Array2<f64>){

    for i in 0..adj.len(){
        let parents = &adj[i];

        let mut parents_d : Vec<usize> = Vec::new();
        for k in parents.clone(){
            parents_d.push( d[k]);
        }

        let p : usize = parents_d.iter().product();

       // println!("{:?}",parents);
        println!("{:?}",p);
       // let param: array![params.slice(s![..,..,i])];
   //     let parents_d:  array![d.slice(s![parents])];
    //    p = parents_dims.product() as usize;
     //   let cim = create_CIM(dims[i],p,param[0], param[1]);

      //  create_node(i,dims[i],param,parents,parents_d,cim);
    }
}

fn create_node(index: usize, d: usize, params: Array2::<f64>, parents: Vec<u32>, parents_d: Vec<usize>, cim: CIM) -> NODE {
    let p : usize = parents_d.iter().product() ;
    let cim = create_CIM(d,p,params[[index,0]], params[[index,1]]);

    NODE {
        index : index,
        d: d,
        params: params,
        parents: parents,
        parents_d : parents_d,
        cim : cim
    }
}

fn create_CIM(d: usize, p:  usize, alpha: f64, beta: f64) -> CIM {
    let gamma = Gamma::new(alpha, beta);
    let mut IM = Array3::<f64>::zeros((d,d,p));

    for u in 0..p {
        for i in 0..d {
            for j in 0..d {
                IM[[i, j, u]] = gamma.ind_sample(&mut rand::thread_rng());
            }
            IM[[i,i,u]] = -IM.slice(s![i,0..i,u]).sum()-IM.slice(s![i,i+1..,u]).sum();
        }
    }
    CIM {
        d: d,
        p: p,
        val : IM
    }
}

fn main() {

    let d = 2;
    let p = 1;

    println!("{:?}",create_CIM(d,p,1.,2.));

    let adj: [Vec<usize>;3] = [vec![1],vec![2],vec![1,2]];
    let d: [usize;3] = [2,3,2];
    let params = Array2::<f64>::ones((2,3));
     create_ctbn(&adj,&d,params);




}