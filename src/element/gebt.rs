#![allow(non_snake_case)]
#![allow(dead_code)]

use core::num;
use std::{
    f64::consts::PI,
    ops::{AddAssign, DivAssign},
};

use itertools::{izip, Itertools};
use nalgebra::{
    DMatrix, DVector, Dyn, Matrix1xX, Matrix3, Matrix3x4, Matrix3xX, Matrix4xX, Matrix6, Matrix6xX,
    MatrixXx3, MatrixXx4, OMatrix, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4,
    Vector6,
};

use super::{
    interp::{gauss_legendre_lobotto_points, lagrange_polynomial, lagrange_polynomial_derivative},
    quadrature::Quadrature,
};

// Shape function matrix [num nodes x num quadrature points]
type MatrixNxQ = OMatrix<f64, Dyn, Dyn>;

//------------------------------------------------------------------------------
// Element
//------------------------------------------------------------------------------

#[derive(Debug)]
pub struct Element {
    pub nodes: Nodes,
    pub qps: Vec<QuadraturePoint>,
    pub shape_func_interp: MatrixNxQ,
    pub shape_func_deriv: MatrixNxQ,
    pub jacobian: Matrix1xX<f64>,
}

#[derive(Debug, Clone)]
pub struct Nodes {
    pub num: usize,
    pub s: DVector<f64>,
    pub xi: DVector<f64>,
    pub x0: Matrix3xX<f64>,
    pub r0: Matrix4xX<f64>,
    pub u: Matrix3xX<f64>,
    pub r: Matrix4xX<f64>,
}

impl Nodes {
    pub fn new(
        s: &DVector<f64>,
        xi: &DVector<f64>,
        x0: &Matrix3xX<f64>,
        r0: &Matrix4xX<f64>,
    ) -> Self {
        let num_nodes = s.len();
        Nodes {
            num: num_nodes,
            s: s.clone(),
            xi: xi.clone(),
            x0: x0.clone(),
            r0: r0.clone(),
            u: Matrix3xX::zeros(num_nodes),
            r: Matrix4xX::zeros(num_nodes),
        }
    }
    pub fn element(&self, quadrature: &Quadrature, sections: &[Section]) -> Element {
        let qp_xi: DVector<f64> = quadrature.points.clone();
        let qp_weights: DVector<f64> = quadrature.weights.clone();
        let num_qp = qp_xi.len();

        // Calculate shape functions weights to interpolate from nodes to quadrature points [nq x nn]
        let shape_func_interp: MatrixNxQ = DMatrix::from_iterator(
            self.num,
            num_qp,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial(xi, self.xi.as_slice())),
        );

        // Calculate shape functions weights to take derivative from nodes to quadrature points [nq x nn]
        let shape_func_deriv: MatrixNxQ = DMatrix::from_iterator(
            self.num,
            num_qp,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial_derivative(xi, self.xi.as_slice())),
        );

        // Get section locations as xi
        let section_xi: Vec<f64> = sections
            .iter()
            .map(|section| (section.s + 1.) / 2.)
            .collect();

        // Calculate the interpolated mass matrix at each quadrature point
        let qp_mass: Vec<Matrix6<f64>> = qp_xi
            .iter()
            .map(|&xi| {
                let weights = lagrange_polynomial(xi, &section_xi);
                sections
                    .iter()
                    .zip(weights.iter())
                    .map(|(section, &w)| section.material.mass * w)
                    .sum()
            })
            .collect();

        // Calculate the interpolated stiffness matrix at each quadrature point
        let qp_stiffness: Vec<Matrix6<f64>> = qp_xi
            .iter()
            .map(|&xi| {
                let weights = lagrange_polynomial(xi, &section_xi);
                sections
                    .iter()
                    .zip(weights.iter())
                    .map(|(section, &w)| section.material.stiffness * w)
                    .sum()
            })
            .collect();

        // Get derivative of node positions wrt xi at the quadrature points
        let qp_deriv: Matrix3xX<f64> = &self.x0 * &shape_func_deriv;

        // Calculate Jacobian matrix
        let jacobian: Matrix1xX<f64> =
            Matrix1xX::from_iterator(qp_xi.len(), qp_deriv.column_iter().map(|r| r.norm()));

        // Interpolate the position, rotation from nodes to the quadrature points
        let qp_x0: Matrix3xX<f64> = interp_position(&shape_func_interp, &self.x0);
        let qp_r0: Matrix4xX<f64> = interp_rotation(&shape_func_interp, &self.r0);

        // Convert rotation matrix to unit quaternions
        let qp_R0: Vec<UnitQuaternion<f64>> = qp_r0
            .column_iter()
            .map(|r| Vector4::from(r).as_unit_quaternion())
            .collect();

        // Calculate the tangent vector of the position at each quadrature point
        let qp_x0_prime: Matrix3xX<f64> =
            interp_position_derivative(&shape_func_deriv, &jacobian, &self.x0);

        let qps: Vec<QuadraturePoint> = izip!(
            qp_xi.iter(),
            qp_x0.column_iter(),
            qp_x0_prime.column_iter(),
            qp_R0.iter(),
            qp_mass.iter(),
            qp_stiffness.iter(),
            qp_weights.iter(),
        )
        .map(
            |(&xi, x0, x0p, &R0, &M_star, &C_star, &w)| QuadraturePoint {
                xi,
                x0: x0.into(),
                x0_prime: x0p.into(),
                R0,
                M_star,
                C_star,
                weight: w,
                ..Default::default()
            },
        )
        .collect();

        // Build and return element
        Element {
            nodes: self.clone(),
            qps,
            shape_func_interp,
            shape_func_deriv,
            jacobian,
        }
    }
}

impl Element {
    pub fn displace(&mut self, node_u: &Matrix3xX<f64>, node_r: &Matrix4xX<f64>) {
        // Interpolate displacement/rotation and derivatives to quadrature points
        let u: Matrix3xX<f64> = interp_position(&self.shape_func_interp, &node_u);
        let u_prime: Matrix3xX<f64> =
            interp_position_derivative(&self.shape_func_deriv, &self.jacobian, &node_u);
        let r: Matrix4xX<f64> = interp_rotation(&self.shape_func_interp, &node_r);
        let r_prime: Matrix4xX<f64> =
            interp_rotation_derivative(&self.shape_func_deriv, &self.jacobian, &node_r);

        // Calculate displaced quadrature point values
        for (i, mut qp) in self.qps.iter_mut().enumerate() {
            let R = Vector4::from(r.column(i)).as_unit_quaternion();
            let Rp = Vector4::from(r_prime.column(i)).as_quaternion();
            qp.displace(
                &Vector3::from(u.column(i)),
                &Vector3::from(u_prime.column(i)),
                &R,
                &Rp,
            )
        }
    }
    pub fn stiffness_matrix(&self) -> DMatrix<f64> {
        let mut K: DMatrix<f64> = DMatrix::zeros(6 * self.nodes.num, 6 * self.nodes.num);
        for i in 0..self.nodes.num {
            for j in 0..self.nodes.num {
                let mut Kij = K.fixed_view_mut::<6, 6>(i * 6, j * 6);
                for (sl, qp) in self.qps.iter().enumerate() {
                    Kij.add_assign(
                        qp.weight
                            * (self.shape_func_interp[(i, sl)]
                                * qp.P
                                * self.shape_func_deriv[(j, sl)]
                                + self.shape_func_interp[(i, sl)]
                                    * qp.Q
                                    * self.shape_func_interp[(j, sl)]
                                    * self.jacobian[sl]
                                + self.shape_func_deriv[(i, sl)]
                                    * qp.C
                                    * self.shape_func_deriv[(j, sl)]
                                    / self.jacobian[sl]
                                + self.shape_func_deriv[(i, sl)]
                                    * qp.O
                                    * self.shape_func_interp[(j, sl)]),
                    );
                }
            }
        }
        K
    }
    pub fn residual_vector(&self) -> DVector<f64> {
        // Calculate residual matrix
        let mut Residual: Matrix6xX<f64> = Matrix6xX::zeros(self.nodes.num);
        for (i, mut col) in Residual.column_iter_mut().enumerate() {
            for (sl, qp) in self.qps.iter().enumerate() {
                col.add_assign(
                    (self.shape_func_deriv[(i, sl)] * qp.Fc
                        + self.jacobian[sl] * self.shape_func_interp[(i, sl)] * qp.Fd)
                        * qp.weight,
                );
            }
        }
        DVector::from_column_slice(Residual.as_slice())
    }
    pub fn constraint_residual_vector(&self) -> Vector6<f64> {
        let r = Vector4::from(self.nodes.r0.column(0)).as_unit_quaternion();
        let logmap: Vector3<f64> = r.scaled_axis();
        Vector6::new(
            self.nodes.u[0],
            self.nodes.u[1],
            self.nodes.u[2],
            logmap[0],
            logmap[1],
            logmap[2],
        )
    }
    pub fn constraints_gradient_matrix(&self) -> Matrix6xX<f64> {
        let mut B = Matrix6xX::zeros(self.nodes.num * 6);
        B.fill_diagonal(1.);
        B
    }
}

// QuadraturePoint defines the quadrature point data in the reference configuration
#[derive(Debug, Default)]
pub struct QuadraturePoint {
    xi: f64,
    x0: Vector3<f64>,
    x0_prime: Vector3<f64>,
    R0: UnitQuaternion<f64>,
    M_star: Matrix6<f64>,
    C_star: Matrix6<f64>,
    weight: f64,
    // Displaced
    u: Vector3<f64>,
    u_prime: Vector3<f64>,
    R: UnitQuaternion<f64>,
    R_prime: Quaternion<f64>,
    RR0: Rotation3<f64>,
    strain: Vector6<f64>,
    M: Matrix6<f64>, // Mass matrix
    C: Matrix6<f64>, // Stiffness matrix
    Fc: Vector6<f64>,
    Fd: Vector6<f64>,
    O: Matrix6<f64>,
    P: Matrix6<f64>,
    Q: Matrix6<f64>,
}

impl QuadraturePoint {
    pub fn displace(
        &mut self,
        u: &Vector3<f64>,
        u_prime: &Vector3<f64>,
        R: &UnitQuaternion<f64>,
        R_prime: &Quaternion<f64>,
    ) {
        // Update displacement inputs
        self.u = *u;
        self.u_prime = *u_prime;
        self.R = *R;
        self.R_prime = *R_prime;

        // Calculate 6x6 rotation matrix
        self.RR0 = (R * self.R0).to_rotation_matrix();
        let mut RR0e: Matrix6<f64> = Matrix6::from_element(0.);
        RR0e.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.RR0.matrix());
        RR0e.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(self.RR0.matrix());

        // Sectional mass matrix
        self.M = RR0e * self.M_star * RR0e.transpose();

        // Sectional stiffness matrix
        self.C = RR0e * self.C_star * RR0e.transpose();
        let C11 = self.C.fixed_view::<3, 3>(0, 0);
        let C21 = self.C.fixed_view::<3, 3>(3, 0);
        let C12 = self.C.fixed_view::<3, 3>(0, 3);

        // Sectional strain
        let e1: Vector3<f64> = self.x0_prime + u_prime - R * self.x0_prime;
        let e2: Vector3<f64> =
            2. * R.F() * Vector4::new(R_prime.w, R_prime.i, R_prime.j, R_prime.k); // kappa
        self.strain = Vector6::new(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]);

        // Calculate skew_symmetric_matrix(x0_prime + u_prime)
        let x0pupSS: Matrix3<f64> = (self.x0_prime + u_prime).skew_symmetric_matrix();

        // Elastic force C
        self.Fc = self.C * self.strain;
        let Nt: Matrix3<f64> =
            Vector3::new(self.Fc[0], self.Fc[1], self.Fc[2]).skew_symmetric_matrix();
        let Mt: Matrix3<f64> =
            Vector3::new(self.Fc[3], self.Fc[4], self.Fc[5]).skew_symmetric_matrix();

        // Elastic force D
        self.Fd.fill(0.);
        self.Fd
            .fixed_rows_mut::<3>(3)
            .copy_from(&(x0pupSS.transpose() * self.Fc.xyz()));

        // Linearization matrices
        self.O.fill(0.);
        self.O
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-Nt + C11 * x0pupSS));
        self.O
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-Mt + C21 * x0pupSS));

        self.P.fill(0.);
        self.P
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(Nt + x0pupSS.transpose() * C11));
        self.P
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * C12));

        self.Q.fill(0.);
        self.Q
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * (-Nt + C11 * x0pupSS)));
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use itertools::Itertools;
    use nalgebra::{Matrix3xX, Matrix4xX};

    use super::*;

    #[test]
    pub fn test_beam3() {
        let fz = |t: f64| -> f64 { t - 2. * t * t };
        let fy = |t: f64| -> f64 { -2. * t + 3. * t * t };
        let fx = |t: f64| -> f64 { 5. * t };
        let ft = |t: f64| -> f64 { 0. * t * t };

        let scale = 0.1;
        let ux = |t: f64| -> f64 { scale * t * t };
        let uy = |t: f64| -> f64 { scale * (t * t * t - t * t) };
        let uz = |t: f64| -> f64 { scale * (t * t + 0.2 * t * t * t) };
        let rot = |t: f64| -> Matrix3<f64> {
            Matrix3::new(
                1.,
                0.,
                0.,
                0.,
                (scale * t).cos(),
                -(scale * t).sin(),
                0.,
                (scale * t).sin(),
                (scale * t).cos(),
            )
        };

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let xi: DVector<f64> = DVector::from_vec(gauss_legendre_lobotto_points(4));
        let s: DVector<f64> = xi.add_scalar(1.) / 2.;

        let x0 = Matrix3xX::from_columns(
            s.iter()
                .map(|&si| Vector3::new(fx(si), fy(si), fz(si)))
                .collect_vec()
                .as_slice(),
        );

        let r0 = Matrix4xX::from_column_slice(&vec![
            0.9778215200524469, // column 1
            -0.01733607539094763,
            -0.09001900002195001,
            -0.18831121859148398,
            0.9950113028068008, // column 2
            -0.002883848832932071,
            -0.030192109815745303,
            -0.09504013471947484,
            0.9904718430204884, // column 3
            -0.009526411091536478,
            0.09620741150793366,
            0.09807604012323785,
            0.9472312341234699, // column 4
            -0.04969214162931507,
            0.18127630174800594,
            0.25965858850765167,
            0.9210746582719719, // column 5
            -0.07193653093139739,
            0.20507529985516368,
            0.32309554437664584,
        ]);

        let nodes = Nodes::new(&s, &xi, &x0, &r0);

        // Create material
        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_fn(|i, j| ((i + 1) * (j + 1)) as f64),
        };

        // Create quadrature points and weights
        let gq = Quadrature::gauss(7);

        // Create element from nodes
        let mut elem = nodes.element(&gq, &vec![Section::new(0.0, &mat), Section::new(1.0, &mat)]);

        // Get xyz displacements at node locations
        let u: Matrix3xX<f64> = Matrix3xX::from_columns(
            s.iter()
                .map(|&si| Vector3::new(ux(si), uy(si), uz(si)))
                .collect_vec()
                .as_slice(),
        );

        // Get vector of unit quaternions describing the nodal rotation
        let r: Matrix4xX<f64> = Matrix4xX::from_columns(
            s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).to_vector())
                .collect_vec()
                .as_slice(),
        );

        // Displace the element
        elem.displace(&u, &r);

        assert_relative_eq!(
            elem.qps[0].u,
            Vector3::new(0.0000647501, -0.0000631025, 0.0000650796),
            epsilon = 1.0e-8
        );

        assert_relative_eq!(
            elem.qps[0].R.to_vector(),
            Vector4::new(0.999999, 0.0012723, 0., 0.),
            epsilon = 1.0e-6
        );

        assert_relative_eq!(
            elem.qps[0].u_prime,
            Vector3::new(0.000941488, -0.000905552, 0.000948675),
            epsilon = 1.0e-8
        );

        assert_relative_eq!(elem.qps[0].R_prime.w, -0.0000117686, epsilon = 1.0e-6);
        assert_relative_eq!(elem.qps[0].R_prime.i, 0.00924984, epsilon = 1.0e-6);
        assert_relative_eq!(elem.qps[0].R_prime.j, 0., epsilon = 1.0e-6);
        assert_relative_eq!(elem.qps[0].R_prime.k, 0., epsilon = 1.0e-6);

        assert_relative_eq!(
            elem.qps[0].strain,
            Vector6::new(0.000941488, -0.000483829, 0.00181883, 0.0184997, 0., 0.),
            epsilon = 1.0e-6
        );

        assert_relative_eq!(
            elem.qps[0].Fc,
            Vector6::new(0.10234, 0.151237, 0.278871, 0.400353, 0.324973, 0.587574),
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            elem.qps[0].Fd,
            Vector6::new(0., 0., 0., 0.120831, 0.241111, -0.175102),
            epsilon = 1.0e-5
        );

        // Get residual vector
        let R: DVector<f64> = elem.residual_vector();
        assert_relative_eq!(
            R,
            DVector::from_vec(vec![
                -0.11121183449279282,
                -0.1614948289968802,
                -0.30437442031624984,
                -0.4038524317172834,
                -0.2927535433573449,
                -0.6838427114868945,
                -0.05271722510789774,
                -0.09881895866851105,
                -0.10322757369767002,
                -0.04702535280785902,
                0.20030704028671845,
                -0.4938097677451336,
                0.11947220141210127,
                0.14340660096243135,
                0.27892509420583494,
                0.2881567336891579,
                0.9034846722720415,
                0.21303887057614374,
                0.08548666343411272,
                0.2979178425020643,
                0.30730427651111003,
                0.3462309471335357,
                0.7532480845688474,
                0.5928285701260345,
                -0.04102980524552408,
                -0.18101065579910575,
                -0.17862737670302692,
                -0.06307503428117978,
                -0.5619802667455867,
                -0.26001250124380276
            ]),
            epsilon = 1.0e-7
        );

        // Get stiffness matrix
        let K: DMatrix<f64> = elem.stiffness_matrix();
        assert_relative_eq!(K[(0, 0)], 1.7424, epsilon = 1.0e-4);
        assert_relative_eq!(K[(0, 1)], 2.5965, epsilon = 1.0e-4);
    }
}

#[derive(Clone)]
pub struct Section {
    pub s: f64,             // Distance along centerline from first point
    pub material: Material, // Material specification
}

impl Section {
    pub fn new(s: f64, material: &Material) -> Self {
        Section {
            s,
            material: material.clone(),
        }
    }
}

#[derive(Clone)]
pub struct Material {
    pub eta: Vector3<f64>,       // Centerline offset
    pub mass: Matrix6<f64>,      // Mass matrix
    pub stiffness: Matrix6<f64>, // Stiffness matrix
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

fn interp_position(
    shape_func_interp: &DMatrix<f64>,
    node_position: &Matrix3xX<f64>,
) -> Matrix3xX<f64> {
    node_position * shape_func_interp
}

fn interp_rotation(
    shape_func_interp: &DMatrix<f64>,
    node_rotation: &Matrix4xX<f64>,
) -> Matrix4xX<f64> {
    let mut r: Matrix4xX<f64> = node_rotation * shape_func_interp;
    for mut c in r.column_iter_mut() {
        c.normalize_mut();
    }
    r
}

fn interp_position_derivative(
    shape_func_deriv: &DMatrix<f64>,
    jacobian: &Matrix1xX<f64>,
    node_position: &Matrix3xX<f64>,
) -> Matrix3xX<f64> {
    let mut u_prime: Matrix3xX<f64> = node_position * shape_func_deriv;
    for mut r in u_prime.row_iter_mut() {
        r.component_div_assign(jacobian);
    }
    u_prime
}

fn interp_rotation_derivative(
    shape_func_deriv: &DMatrix<f64>,
    jacobian: &Matrix1xX<f64>,
    node_rotation: &Matrix4xX<f64>,
) -> Matrix4xX<f64> {
    let mut r_prime: Matrix4xX<f64> = node_rotation * shape_func_deriv;
    for mut r in r_prime.row_iter_mut() {
        r.component_div_assign(jacobian);
    }
    r_prime
}

//------------------------------------------------------------------------------
// Traits
//------------------------------------------------------------------------------

pub trait VecToQuatExt {
    fn as_quaternion(&self) -> Quaternion<f64>;
    fn as_unit_quaternion(&self) -> UnitQuaternion<f64>;
}

impl VecToQuatExt for Vector4<f64> {
    fn as_quaternion(&self) -> Quaternion<f64> {
        Quaternion::new(self[0], self[1], self[2], self[3])
    }
    fn as_unit_quaternion(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(self.as_quaternion())
    }
}

pub trait SkewSymmExt {
    fn skew_symmetric_matrix(&self) -> Matrix3<f64>;
}

impl SkewSymmExt for Vector3<f64> {
    fn skew_symmetric_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            0.0, -self[2], self[1], self[2], 0.0, -self[0], -self[1], self[0], 0.0,
        )
    }
}

pub trait QuatExt {
    fn F(&self) -> Matrix3x4<f64>;
    fn to_vector(&self) -> Vector4<f64>;
}

impl QuatExt for UnitQuaternion<f64> {
    fn F(&self) -> Matrix3x4<f64> {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(-q1, q0, -q3, q2, -q2, q3, q0, -q1, -q3, -q2, q1, q0)
    }
    fn to_vector(&self) -> Vector4<f64> {
        Vector4::new(self.w, self.i, self.j, self.k)
    }
}
