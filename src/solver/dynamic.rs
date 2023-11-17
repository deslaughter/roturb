
use nalgebra::DVector;

pub struct Solver{}

struct State{
    displacement: DVector<f64>,
    velocity: DVector<f64>,
    acceleration: DVector<f64>,
    algorithm_acceleration: DVector<f64>,
}

impl State {
    pub fn new(ndof: usize) -> Self {
        Self { 
            displacement:DVector::from_element(ndof, 0.0),
            velocity:DVector::from_element(ndof, 0.0),
            acceleration:DVector::from_element(ndof, 0.0),
            algorithm_acceleration:DVector::from_element(ndof, 0.0),
         }
    }
}