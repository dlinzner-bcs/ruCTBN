use ndarray::prelude::*;
use ndarray::Array;
use rand::distributions::Gamma;
#[derive(Debug)]

struct CIM {
    d:  usize,
    p:  usize,
    val : Array3::<f64>
}

struct node {
    index: u32,
    cim: CIM,
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
}