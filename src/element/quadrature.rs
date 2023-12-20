use nalgebra::DVector;

use super::interp::gauss_legendre_lobotto_quadrature;

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
    pub fn gauss_legendre_lobotto(order: usize) -> Self {
        let (p, w) = gauss_legendre_lobotto_quadrature(order);
        Quadrature {
            points: DVector::from_vec(p),
            weights: DVector::from_vec(w),
        }
    }
}
