#[macro_use]
use ndarray::prelude::*;
use ndarray::Array;
use rand::distributions;

struct CIM {
    dims: int32,
    pims: int32,
    val : Array::<f64, _>::ones((dims, dims,pims))
}

fn create_CIM(dims: int32, pims: int32, alpha: f64, beta: f64) -> CIM {
    let mut IM = Array::<f64, _>::ones((dims, dims));
    for (i,j) in (5..10).enumerate(){
            IM[[i,j]] = 1;
    }

    CIM {
        dims: dims,
        pims: pims,

    }
}


fn main() {
    let a = arr2(&[[1.,2.,3.], [4.,5.,6.]]);
    let mut b = Array::<f64, _>::ones((2, 3).f());

    b[[0,0]]= b[[0,0]]+1.;

    println!("{}",a+b);
}