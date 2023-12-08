use nalgebra::DVector;

pub struct Quadrature {
    pub points: DVector<f64>,
    pub weights: DVector<f64>,
}

impl Quadrature {
    pub fn gauss(order: usize) -> Self {
        let gl_rule = gauss_quad::GaussLegendre::init(order);
        Quadrature {
            points: DVector::from_iterator(order, gl_rule.nodes.into_iter().rev()),
            weights: DVector::from_iterator(order, gl_rule.weights.into_iter().rev()),
        }
    }
}
