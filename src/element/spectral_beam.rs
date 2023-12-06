#![allow(non_snake_case)]
#![allow(dead_code)]

use std::{f64::consts::PI, ops::AddAssign};

use itertools::{izip, Itertools};
use nalgebra::{
    distance, DMatrix, DVector, Matrix3, Matrix3x4, Matrix3xX, Matrix6, Matrix6xX, MatrixXx1,
    MatrixXx3, MatrixXx4, Point4, Quaternion, Rotation3, RowVector4, UnitQuaternion, Vector3,
    Vector4, Vector6,
};

use super::interp::{
    gauss_legendre_lobotto_points, lagrange_polynomial, lagrange_polynomial_derivative,
    quaternion_from_tangent_twist,
};

//------------------------------------------------------------------------------
// Beam
//------------------------------------------------------------------------------

struct Beam {
    length: f64,
    ref_position: Matrix3xX<f64>, // x, y, z
    ref_twist: DVector<f64>,
    ref_locs: DVector<f64>,
    sections: Vec<Section>,
    section_locs: Vec<f64>,
}

impl Beam {
    fn new(points: &[Point4<f64>], sections: &[Section]) -> Self {
        let dist = DVector::from_iterator(
            points.len() - 1,
            points
                .windows(2)
                .map(|s| distance(&s[0].xyz(), &s[1].xyz())),
        );
        Beam {
            length: dist.sum(),
            ref_position: Matrix3xX::from_columns(
                &points
                    .iter()
                    .map(|&p| p.xyz().coords)
                    .collect::<Vec<Vector3<f64>>>(),
            ),
            ref_twist: DVector::from_iterator(points.len(), points.iter().map(|&p| p.w)),
            ref_locs: DVector::from_iterator(
                points.len(),
                (0..1)
                    .into_iter()
                    .map(|v| v as f64)
                    .chain(dist.iter().scan(0.0, |acc, &x| {
                        *acc += x;
                        Some(*acc)
                    })),
            ),
            sections: sections.to_vec(),
            section_locs: sections.iter().map(|s| s.s).collect(),
        }
    }

    fn element(&self, node_order: usize, quadrature_order: usize, loc1: f64, loc2: f64) -> Element {
        let node_xi = DVector::from(gauss_legendre_lobotto_points(node_order));
        let node_locs = node_xi
            .add_scalar(1.)
            .scale((loc2 - loc1) / 2.)
            .add_scalar(loc1);

        // Get weights for each node location to go from reference data to node value
        let node_interp_weight = DMatrix::from_row_iterator(
            node_locs.len(),
            self.ref_locs.len(),
            node_locs
                .iter()
                .flat_map(|&loc| lagrange_polynomial(loc, self.ref_locs.as_slice())),
        );

        // Get weights for each node location to go from reference data to node value
        let node_deriv_weight = DMatrix::from_row_iterator(
            node_locs.len(),
            self.ref_locs.len(),
            node_locs
                .iter()
                .flat_map(|&loc| lagrange_polynomial_derivative(loc, self.ref_locs.as_slice())),
        );

        // Calculate initial rotation at each node
        // let qp_rotations: Vec<UnitQuaternion<f64>> = qp_x0p
        //     .row_iter()
        //     .zip(qp_twists.row_iter())
        //     .map(|(deriv, twist)| {
        //         let tangent = deriv.normalize();
        //         quaternion_from_tangent_twist(&tangent.transpose(), twist.x)
        //     })
        //     .collect();

        // Calculate position, derivatives, and twist at all nodes
        let node_positions: MatrixXx3<f64> = &node_interp_weight * &self.ref_position.transpose();
        let node_derivs: MatrixXx3<f64> = &node_deriv_weight * &self.ref_position.transpose();
        let node_twists: MatrixXx1<f64> = &node_interp_weight * &self.ref_twist;

        // Calculate initial rotation at each node
        let node_rotations: MatrixXx4<f64> = MatrixXx4::from_rows(
            &(node_derivs
                .row_iter()
                .zip(node_twists.row_iter())
                .map(|(tangent, twist)| {
                    let e1: Vector3<f64> = tangent.normalize().transpose();
                    let a = if e1[0] > 0. { 1. } else { -1. };
                    let e2: Vector3<f64> = Vector3::new(
                        -a * e1[1] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
                        a * e1[0] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
                        0.,
                    );
                    let e3: Vector3<f64> = e1.cross(&e2);
                    let r_tan: Rotation3<f64> =
                        Rotation3::from_matrix(&Matrix3::from_columns(&[e1, e2, e3]));
                    let r_twist: Rotation3<f64> =
                        Rotation3::from_scaled_axis(e1 * twist * PI / 180.);
                    let q = UnitQuaternion::from_rotation_matrix(&(r_twist * r_tan));
                    RowVector4::new(q.w, q.i, q.j, q.k)
                })
                .collect::<Vec<RowVector4<f64>>>()),
        );

        // Create nodes structure
        let nodes = Nodes {
            num: node_xi.len(),
            locs: node_locs,
            xi: node_xi,
            x0: node_positions,
            r0: node_rotations,
        };

        // Get element from nodes
        nodes.element(quadrature_order, &self.sections)
    }
}

//------------------------------------------------------------------------------
// Element
//------------------------------------------------------------------------------

#[derive(Debug)]
struct Element {
    nodes: Nodes,
    qps: Vec<QuadraturePoint>,
    shape_func_interp: DMatrix<f64>,
    shape_func_deriv: DMatrix<f64>,
    jacobian: DVector<f64>,
}

struct ElementDisp {
    qps: Vec<QuadraturePointsDisp>,
}

#[derive(Debug)]
pub struct QuadraturePoints {
    num: usize,
    xi: DVector<f64>,
    x0: MatrixXx3<f64>,
    x0_prime: MatrixXx3<f64>,
    r0: MatrixXx4<f64>,
    M_star: Vec<Matrix6<f64>>,
    C_star: Vec<Matrix6<f64>>,
    weight: DVector<f64>,
}

#[derive(Debug, Clone)]
pub struct Nodes {
    num: usize,
    locs: DVector<f64>,
    xi: DVector<f64>,
    x0: MatrixXx3<f64>,
    r0: MatrixXx4<f64>,
}

impl Nodes {
    fn element(&self, order: usize, sections: &[Section]) -> Element {
        let gl_rule = gauss_quad::GaussLegendre::init(order);
        let qp_xi = DVector::from_iterator(order, gl_rule.nodes.into_iter().rev());
        let qp_weights = DVector::from_iterator(order, gl_rule.weights.into_iter().rev());
        let num_qp = qp_xi.len();

        // Calculate shape functions weights to interpolate from nodes to quadrature points [nq x nn]
        let shape_func_interp: DMatrix<f64> = DMatrix::from_row_iterator(
            num_qp,
            self.num,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial(xi, self.xi.as_slice())),
        );

        // Calculate shape functions weights to take derivative from nodes to quadrature points [nq x nn]
        let shape_func_deriv: DMatrix<f64> = DMatrix::from_row_iterator(
            num_qp,
            self.num,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial_derivative(xi, self.xi.as_slice())),
        );

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
        let qp_deriv: MatrixXx3<f64> = &shape_func_deriv * &self.x0;

        // Calculate Jacobian matrix
        let jacobian: DVector<f64> =
            DVector::from_iterator(qp_xi.len(), qp_deriv.row_iter().map(|r| r.norm()));

        // Interpolate the position, rotation from nodes to the quadrature points
        let qp_x0: MatrixXx3<f64> = interp_position(&shape_func_interp, &self.x0);
        let qp_r0: MatrixXx4<f64> = interp_rotation(&shape_func_interp, &self.r0);

        // Convert rotation matrix to unit quaternions
        let qp_R0: Vec<UnitQuaternion<f64>> = qp_r0
            .row_iter()
            .map(|r| UnitQuaternion::from_quaternion(Quaternion::new(r[0], r[1], r[2], r[3])))
            .collect();

        // Calculate the tangent vector of the position at each quadrature point
        let qp_x0_prime: MatrixXx3<f64> =
            interp_position_derivative(&shape_func_deriv, &jacobian, &self.x0);

        let qps: Vec<QuadraturePoint> = izip!(
            qp_xi.iter(),
            qp_x0.row_iter(),
            qp_x0_prime.row_iter(),
            qp_R0.iter(),
            qp_mass.iter(),
            qp_stiffness.iter(),
            qp_weights.iter(),
        )
        .map(
            |(&xi, x0, x0p, &R0, &M_star, &C_star, &w)| QuadraturePoint {
                xi,
                x0: x0.transpose(),
                x0_prime: x0p.transpose(),
                R0,
                M_star,
                C_star,
                weight: w,
            },
        )
        .collect();

        // Build and return element
        Element {
            nodes: self.clone(),
            qps: qps,
            shape_func_interp,
            shape_func_deriv,
            jacobian,
        }
    }
}

fn interp_position(
    shape_func_interp: &DMatrix<f64>,
    node_position: &MatrixXx3<f64>,
) -> MatrixXx3<f64> {
    shape_func_interp * node_position
}

fn interp_rotation(
    shape_func_interp: &DMatrix<f64>,
    node_rotation: &MatrixXx4<f64>,
) -> MatrixXx4<f64> {
    MatrixXx4::from_rows(
        &((shape_func_interp * node_rotation)
            .row_iter()
            .map(|r| r.normalize())
            .collect::<Vec<RowVector4<f64>>>()),
    )
}

fn interp_position_derivative(
    shape_func_deriv: &DMatrix<f64>,
    jacobian: &DVector<f64>,
    node_position: &MatrixXx3<f64>,
) -> MatrixXx3<f64> {
    MatrixXx3::from_columns(
        &(shape_func_deriv * node_position)
            .column_iter()
            .map(|c| c.component_div(jacobian))
            .collect::<Vec<DVector<f64>>>(),
    )
}

fn interp_rotation_derivative(
    shape_func_deriv: &DMatrix<f64>,
    jacobian: &DVector<f64>,
    node_rotation: &MatrixXx4<f64>,
) -> MatrixXx4<f64> {
    MatrixXx4::from_columns(
        &(shape_func_deriv * node_rotation)
            .column_iter()
            .map(|c| c.component_div(jacobian))
            .collect::<Vec<DVector<f64>>>(),
    )
}

// QuadraturePointRef defines the quadrature point data in the reference configuration
#[derive(Debug)]
pub struct QuadraturePoint {
    xi: f64,
    x0: Vector3<f64>,
    x0_prime: Vector3<f64>,
    R0: UnitQuaternion<f64>,
    M_star: Matrix6<f64>,
    C_star: Matrix6<f64>,
    weight: f64,
}

impl Element {
    fn displace(&self, node_u: &MatrixXx3<f64>, node_r: &MatrixXx4<f64>) -> ElementDisp {
        // Interpolate displacement/rotation and derivatives to quadrature points
        let u: MatrixXx3<f64> = interp_position(&self.shape_func_interp, &node_u);
        let u_prime: MatrixXx3<f64> =
            interp_position_derivative(&self.shape_func_deriv, &self.jacobian, &node_u);
        let r: MatrixXx4<f64> = interp_rotation(&self.shape_func_interp, &node_r);
        let r_prime: MatrixXx4<f64> =
            interp_rotation_derivative(&self.shape_func_deriv, &self.jacobian, &node_r);

        // Convert rotation and derivative to quaternions
        let R: Vec<UnitQuaternion<f64>> = r
            .row_iter()
            .map(|r| UnitQuaternion::from_quaternion(Quaternion::new(r[0], r[1], r[2], r[3])))
            .collect();
        let Rp: Vec<Quaternion<f64>> = r_prime
            .row_iter()
            .map(|r| Quaternion::new(r[0], r[1], r[2], r[3]))
            .collect();

        // Calculate displaced quadrature point data
        let qpds: Vec<QuadraturePointsDisp> = izip!(
            self.qps.iter(),
            u.transpose().column_iter(),
            u_prime.transpose().column_iter(),
            R.iter(),
            Rp.iter()
        )
        .map(|(qp, u, u_prime, R, R_prime)| {
            qp.displace(&Vector3::from(u), &Vector3::from(u_prime), R, R_prime)
        })
        .collect();

        // Calculate residual matrix
        let mut Residual: Matrix6xX<f64> = Matrix6xX::zeros(self.nodes.num);
        for (i, mut col) in Residual.column_iter_mut().enumerate() {
            let phi_i = self.shape_func_interp.column(i);
            let phip_i = self.shape_func_deriv.column(i);
            for (sl, qp) in qpds.iter().enumerate() {
                col.add_assign(
                    (phip_i[sl] * qp.Fc + self.jacobian[sl] * phi_i[sl] * qp.Fd)
                        * self.qps[sl].weight,
                );
            }
        }

        // Calculate stiffness matrix
        let mut K: DMatrix<f64> = DMatrix::zeros(6 * self.nodes.num, 6 * self.nodes.num);
        let phi = &self.shape_func_interp;
        let phip = &self.shape_func_deriv;
        for i in 0..self.nodes.num {
            for j in 0..self.nodes.num {
                let mut Kij = K.fixed_view_mut::<6, 6>(i * 6, j * 6);
                for (sl, qp) in qpds.iter().enumerate() {
                    Kij.add_assign(
                        self.qps[sl].weight
                            * (phi[(sl, i)] * qp.P * phip[(sl, j)]
                                + phi[(sl, i)] * qp.Q * phi[(sl, j)] * self.jacobian[sl]
                                + phip[(sl, i)] * qp.C * phip[(sl, j)] / self.jacobian[sl]
                                + phip[(sl, i)] * qp.O * phi[(sl, j)]),
                    );
                }
            }
        }

        ElementDisp { qps: qpds }
    }
}

impl QuadraturePoint {
    fn displace(
        &self,
        u: &Vector3<f64>,
        u_prime: &Vector3<f64>,
        R: &UnitQuaternion<f64>,
        R_prime: &Quaternion<f64>,
    ) -> QuadraturePointsDisp {
        // Calculate 6x6 rotation matrix
        let RR0 = (R * self.R0).to_rotation_matrix();
        let mut RR0e: Matrix6<f64> = Matrix6::from_element(0.);
        RR0e.fixed_view_mut::<3, 3>(0, 0).copy_from(RR0.matrix());
        RR0e.fixed_view_mut::<3, 3>(3, 3).copy_from(RR0.matrix());

        // Sectional mass matrix
        let M: Matrix6<f64> = RR0e * self.M_star * RR0e.transpose();

        // Sectional stiffness matrix
        let C: Matrix6<f64> = RR0e * self.C_star * RR0e.transpose();
        let C11 = C.fixed_view::<3, 3>(0, 0);
        let C21 = C.fixed_view::<3, 3>(3, 0);
        let C12 = C.fixed_view::<3, 3>(0, 3);

        // Sectional strain
        let e1: Vector3<f64> = self.x0_prime + u_prime - R * self.x0_prime;
        let e2: Vector3<f64> =
            2. * R.F() * Vector4::new(R_prime.w, R_prime.i, R_prime.j, R_prime.k); // kappa
        let strain: Vector6<f64> = Vector6::new(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]);

        // Calculate skew_symmetric_matrix(x0_prime + u_prime)
        let x0pupSS: Matrix3<f64> = (self.x0_prime + u_prime).skew_symmetric_matrix();

        // Elastic force C
        let Fc: Vector6<f64> = C * strain;
        let Nt: Matrix3<f64> = Vector3::new(Fc[0], Fc[1], Fc[2]).skew_symmetric_matrix();
        let Mt: Matrix3<f64> = Vector3::new(Fc[3], Fc[4], Fc[5]).skew_symmetric_matrix();

        // Elastic force D
        let mut Fd: Vector6<f64> = Vector6::from_element(0.0);
        Fd.fixed_rows_mut::<3>(3)
            .copy_from(&(x0pupSS.transpose() * Fc.xyz()));

        // Linearization matrices
        let mut O: Matrix6<f64> = Matrix6::from_element(0.0);
        O.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-Nt + C11 * x0pupSS));
        O.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-Mt + C21 * x0pupSS));

        let mut P: Matrix6<f64> = Matrix6::from_element(0.0);
        P.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(Nt + x0pupSS.transpose() * C11));
        P.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * C12));

        let mut Q: Matrix6<f64> = Matrix6::from_element(0.0);
        Q.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * (-Nt + C11 * x0pupSS)));

        // Displaced quadrature point
        QuadraturePointsDisp {
            R: *R,
            R_prime: *R_prime,
            RR0,
            M,
            C,
            u_prime: *u_prime,
            strain,
            O,
            P,
            Q,
            Fc,
            Fd,
        }
    }
}

// QuadraturePointDisp defines the quadrature point data in the displaced/deformed configuration
pub struct QuadraturePointsDisp {
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

fn kappa(q: &UnitQuaternion<f64>, q_prime: &Quaternion<f64>) -> Vector3<f64> {
    2. * q.F() * Vector4::new(q_prime.w, q_prime.i, q_prime.j, q_prime.k)
}

fn omega(q: &UnitQuaternion<f64>, q_dot: &Quaternion<f64>) -> Vector3<f64> {
    2. * q.F() * Vector4::new(q_dot.w, q_dot.i, q_dot.j, q_dot.k)
}

#[cfg(test)]
mod tests {

    use nalgebra::RowVector3;

    use super::*;

    #[test]
    fn test_beam3() {
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

        let nodes = Nodes {
            num: s.len(),
            locs: s.clone(),
            xi: xi,
            x0: MatrixXx3::from_rows(
                &s.iter()
                    .map(|&si| RowVector3::new(fx(si), fy(si), fz(si)))
                    .collect::<Vec<RowVector3<f64>>>(),
            ),
            r0: MatrixXx4::from_row_slice(&vec![
                0.9778215200524469,
                -0.01733607539094763,
                -0.09001900002195001,
                -0.18831121859148398, // row 1
                0.9950113028068008,
                -0.002883848832932071,
                -0.030192109815745303,
                -0.09504013471947484, // row 2
                0.9904718430204884,
                -0.009526411091536478,
                0.09620741150793366,
                0.09807604012323785, // row 3
                0.9472312341234699,
                -0.04969214162931507,
                0.18127630174800594,
                0.25965858850765167, // row 4
                0.9210746582719719,
                -0.07193653093139739,
                0.20507529985516368,
                0.32309554437664584, // row 5
            ]),
        };

        // Create material
        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_fn(|i, j| ((i + 1) * (j + 1)) as f64),
        };

        // Create element from nodes
        let elem = nodes.element(7, &vec![Section::new(0.0, &mat), Section::new(1.0, &mat)]);

        // Get xyz displacements at node locations
        let u: MatrixXx3<f64> = MatrixXx3::from_rows(
            s.iter()
                .map(|&si| RowVector3::new(ux(si), uy(si), uz(si)))
                .collect::<Vec<RowVector3<f64>>>()
                .as_slice(),
        );

        // Get vector of unit quaternions describing the nodal rotation
        let r: MatrixXx4<f64> = MatrixXx4::from_rows(
            s.iter()
                .map(|&si| {
                    let q = UnitQuaternion::from_matrix(&rot(si));
                    RowVector4::new(q.w, q.i, q.j, q.k)
                })
                .collect::<Vec<RowVector4<f64>>>()
                .as_slice(),
        );

        // Get the deformed element
        let elem_disp = elem.displace(&u, &r);
    }
}

#[derive(Clone)]
struct Section {
    s: f64,             // Distance along centerline from first point
    material: Material, // Material specification
}

impl Section {
    fn new(s: f64, material: &Material) -> Self {
        Section {
            s,
            material: material.clone(),
        }
    }
}

#[derive(Clone)]
struct Material {
    eta: Vector3<f64>,       // Centerline offset
    mass: Matrix6<f64>,      // Mass matrix
    stiffness: Matrix6<f64>, // Stiffness matrix
}

//------------------------------------------------------------------------------
// Traits
//------------------------------------------------------------------------------

trait SkewSymmExt {
    fn skew_symmetric_matrix(&self) -> Matrix3<f64>;
}

impl SkewSymmExt for Vector3<f64> {
    fn skew_symmetric_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            0.0, -self[2], self[1], self[2], 0.0, -self[0], -self[1], self[0], 0.0,
        )
    }
}

trait QuatExt {
    fn F(&self) -> Matrix3x4<f64>;
}

impl QuatExt for UnitQuaternion<f64> {
    fn F(&self) -> Matrix3x4<f64> {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(-q1, q0, -q3, q2, -q2, q3, q0, -q1, -q3, -q2, q1, q0)
    }
}
