#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ops::AddAssign;

use itertools::izip;
use nalgebra::{
    distance, DMatrix, DVector, Matrix3, Matrix3x4, Matrix3xX, Matrix6, Matrix6xX, MatrixXx1,
    MatrixXx3, MatrixXx4, Point4, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4, Vector6,
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
            section_locs: sections.iter().map(|s| s.loc).collect(),
        }
    }

    fn element(&self, node_order: usize, quadrature_order: usize, loc1: f64, loc2: f64) -> Element {
        let node_xi = DVector::from(gauss_legendre_lobotto_points(node_order));
        let node_locs = node_xi
            .add_scalar(1.)
            .scale((loc2 - loc1) / 2.)
            .add_scalar(loc1);

        let gl_rule = gauss_quad::GaussLegendre::init(quadrature_order);
        let qp_xi = DVector::from_iterator(quadrature_order, gl_rule.nodes.into_iter().rev());
        let qp_weights =
            DVector::from_iterator(quadrature_order, gl_rule.weights.into_iter().rev());
        let qp_locs = qp_xi
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

        // Calculate position, derivatives, and twist at all nodes
        let node_positions: MatrixXx3<f64> = &node_interp_weight * &self.ref_position.transpose();
        let node_derivs: MatrixXx3<f64> = &node_deriv_weight * &self.ref_position.transpose();
        let node_twists: MatrixXx1<f64> = &node_interp_weight * &self.ref_twist;

        // Calculate initial rotation at each node
        let node_rotations: Vec<UnitQuaternion<f64>> = node_derivs
            .row_iter()
            .zip(node_twists.row_iter())
            .map(|(dr, twist)| {
                let tangent = dr.normalize();
                quaternion_from_tangent_twist(&tangent.transpose(), twist.x)
            })
            .collect();

        // Get weights for each node location to go from reference data to node value
        let qp_interp_weight = DMatrix::from_row_iterator(
            qp_locs.len(),
            self.ref_locs.len(),
            qp_locs
                .iter()
                .flat_map(|&loc| lagrange_polynomial(loc, self.ref_locs.as_slice())),
        );

        // Get weights for each node location to go from reference data to node value
        let qp_deriv_weight = DMatrix::from_row_iterator(
            qp_locs.len(),
            self.ref_locs.len(),
            qp_locs
                .iter()
                .flat_map(|&loc| lagrange_polynomial_derivative(loc, self.ref_locs.as_slice())),
        );

        // Calculate the interpolated mass matrix at each quadrature point
        let qp_mass: Vec<Matrix6<f64>> = qp_interp_weight
            .row_iter()
            .map(|w_interp| {
                self.sections
                    .iter()
                    .zip(w_interp.iter())
                    .map(|(section, &w)| section.material.mass * w)
                    .sum()
            })
            .collect();

        // Calculate the interpolated stiffness matrix at each quadrature point
        let qp_stiffness: Vec<Matrix6<f64>> = qp_interp_weight
            .row_iter()
            .map(|w_interp| {
                self.sections
                    .iter()
                    .zip(w_interp.iter())
                    .map(|(section, &w)| section.material.stiffness * w)
                    .sum()
            })
            .collect();

        // Interpolate the position, derivative, and twist from nodes to the quadrature points
        let qp_positions: MatrixXx3<f64> = &qp_interp_weight * &self.ref_position.transpose();
        let qp_derivs: MatrixXx3<f64> = &qp_deriv_weight * &self.ref_position.transpose();
        let qp_twists: MatrixXx1<f64> = &qp_interp_weight * &self.ref_twist;

        // Calculate initial rotation at each node
        let qp_rotations: Vec<UnitQuaternion<f64>> = qp_derivs
            .row_iter()
            .zip(qp_twists.row_iter())
            .map(|(deriv, twist)| {
                let tangent = deriv.normalize();
                quaternion_from_tangent_twist(&tangent.transpose(), twist.x)
            })
            .collect();

        // Calculate shape functions weights to interpolate from nodes to quadrature points [nq x nn]
        let shape_func_interp: DMatrix<f64> = DMatrix::from_row_iterator(
            qp_xi.len(),
            node_xi.len(),
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial(xi, node_xi.as_slice())),
        );

        // Calculate shape functions weights to take derivative from nodes to quadrature points [nq x nn]
        let shape_func_deriv: DMatrix<f64> = DMatrix::from_row_iterator(
            qp_xi.len(),
            node_xi.len(),
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial_derivative(xi, node_xi.as_slice())),
        );

        // Get derivative of node positions wrt xi at the quadrature points
        let qp_deriv_shp: MatrixXx3<f64> = &shape_func_deriv * &node_positions;

        // Calculate Jacobian matrix
        let jacobian: MatrixXx1<f64> =
            DVector::from_iterator(qp_xi.len(), qp_deriv_shp.row_iter().map(|r| r.norm()));
        let jacobian: MatrixXx3<f64> =
            MatrixXx3::from_columns(&[jacobian.clone(), jacobian.clone(), jacobian.clone()]);

        // Calculate the tangent vector of the position at each quadrature point
        let qp_deriv: MatrixXx3<f64> = qp_deriv_shp.component_div(&jacobian);

        // Build and return element
        Element {
            nodes: Nodes {
                nn: node_xi.len(),
                xi: node_xi,
                locs: node_locs,
                x0: node_positions,
                R0: node_rotations,
            },
            qps: qp_locs
                .iter()
                .zip(qp_xi.iter())
                .zip(qp_positions.row_iter())
                .zip(qp_deriv.row_iter())
                .zip(qp_rotations.iter())
                .zip(qp_mass.iter())
                .zip(qp_stiffness.iter())
                .zip(qp_weights.iter())
                .map(
                    |(((((((&loc, &xi), x0), x0_prime), &R0), &Mstar), &Cstar), &weight)| {
                        QuadraturePointRef {
                            loc,
                            xi,
                            weight: weight,
                            x0: x0.transpose(),
                            x0_prime: x0_prime.transpose(),
                            R0,
                            Mstar,
                            Cstar,
                        }
                    },
                )
                .collect(),
            shape_func_interp,
            shape_func_deriv,
            jacobian,
        }
    }
}

//------------------------------------------------------------------------------
// Element
//------------------------------------------------------------------------------

#[derive(Debug)]
struct Element {
    nodes: Nodes,
    qps: Vec<QuadraturePointRef>,
    shape_func_interp: DMatrix<f64>,
    shape_func_deriv: DMatrix<f64>,
    jacobian: MatrixXx3<f64>,
}

#[derive(Debug)]
pub struct Nodes {
    nn: usize,
    locs: DVector<f64>,
    xi: DVector<f64>,
    x0: MatrixXx3<f64>,
    R0: Vec<UnitQuaternion<f64>>,
}

// QuadraturePointRef defines the quadrature point data in the reference configuration
#[derive(Debug)]
pub struct QuadraturePointRef {
    loc: f64,
    xi: f64,
    weight: f64,
    x0: Vector3<f64>,
    x0_prime: Vector3<f64>,
    R0: UnitQuaternion<f64>,
    Mstar: Matrix6<f64>,
    Cstar: Matrix6<f64>,
}

impl QuadraturePointRef {
    fn displace(
        &self,
        R: &UnitQuaternion<f64>,
        R_p: &Quaternion<f64>,
        u_p: &Vector3<f64>,
    ) -> QuadraturePointDisp {
        let RR0_ = (R * self.R0).to_rotation_matrix();
        let mut RR0: Matrix6<f64> = Matrix6::from_element(0.);
        RR0.fixed_view_mut::<3, 3>(0, 0).copy_from(RR0_.matrix());
        RR0.fixed_view_mut::<3, 3>(3, 3).copy_from(RR0_.matrix());

        // Sectional mass (M) and stiffness (C) matrices
        let M: Matrix6<f64> = RR0 * self.Mstar * RR0.transpose();
        let C: Matrix6<f64> = RR0 * self.Cstar * RR0.transpose();
        let C11 = C.fixed_view::<3, 3>(0, 0);
        let C21 = C.fixed_view::<3, 3>(3, 0);
        let C12 = C.fixed_view::<3, 3>(0, 3);

        // Sectional strain
        let strain = self.sectional_strain_inertial_basis(&u_p, &R, &R_p);

        // Elastic force C
        let Fc = C * strain;
        let Nt: Matrix3<f64> = Fc.xyz().skew_symmetric_matrix();
        let Mt: Matrix3<f64> = Vector3::new(Fc[3], Fc[4], Fc[5]).skew_symmetric_matrix();
        let x0pu0pSS: Matrix3<f64> =
            &self.x0_prime.skew_symmetric_matrix() + &u_p.skew_symmetric_matrix();

        // Elastic force D
        let mut Fd: Vector6<f64> = Vector6::from_element(0.0);
        Fd.fixed_rows_mut::<3>(3)
            .copy_from(&(x0pu0pSS.transpose() * Fc.xyz()));

        // Linearization matrices
        let mut O: Matrix6<f64> = Matrix6::from_element(0.0);
        O.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-Nt + C11 * x0pu0pSS));
        O.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-Mt + C21 * x0pu0pSS));

        let mut P: Matrix6<f64> = Matrix6::from_element(0.0);
        P.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(Nt + x0pu0pSS.transpose() * C11));
        P.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pu0pSS.transpose() * C12));

        let mut Q: Matrix6<f64> = Matrix6::from_element(0.0);
        Q.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pu0pSS.transpose() * (-Nt + C11 * x0pu0pSS)));

        // Displaced quadrature point
        QuadraturePointDisp {
            R: R.clone(),
            R_prime: R_p.clone(),
            RR0: RR0_,
            M,
            C,
            u_prime: u_p.clone(),
            strain,
            O,
            P,
            Q,
            Fc,
            Fd,
        }
    }
}

impl Element {
    fn displace(&mut self, u_node: &MatrixXx3<f64>, R_node: &[UnitQuaternion<f64>]) -> ElementDisp {
        // Calculate derivative of displacement at each quadrature point
        let u_prime = (&self.shape_func_deriv * u_node).component_div(&self.jacobian);

        // Calculate displacement rotation unit quaternions at quadrature points
        let R = self.interp_node_rotation_to_qp(&R_node);

        // Calculate displacement rotation derivative quaternions at quadrature points
        let R_prime = self.interp_node_rotation_deriv_to_qp(&R_node);

        // Calculate quadrature point data in the displaced configuration
        let qps: Vec<QuadraturePointDisp> = izip!(
            self.qps.iter(),
            R.iter(),
            R_prime.iter(),
            u_prime.transpose().column_iter()
        )
        .map(|(qp, R, R_p, u_p)| qp.displace(R, R_p, &Vector3::from(u_p)))
        .collect();

        // // Calculate forces and moments at nodes
        // let qp_Fc: Matrix6xX<f64> = Matrix6xX::from_columns(
        //     qps.iter()
        //         .map(|qp| qp.Fc)
        //         .collect::<Vec<Vector6<f64>>>()
        //         .as_slice(),
        // );
        // let qp_Fd: Matrix6xX<f64> = Matrix6xX::from_columns(
        //     qps.iter()
        //         .map(|qp| qp.Fd)
        //         .collect::<Vec<Vector6<f64>>>()
        //         .as_slice(),
        // );

        // Get integration weights
        let weights: DVector<f64> =
            DVector::from_vec(self.qps.iter().map(|qp| qp.weight).collect());

        // Calculate interpolation shape function multiplied by the Jacobian and weights
        let mut shape_func_interp_jac_weight = self.shape_func_interp.clone();
        for mut c in shape_func_interp_jac_weight.column_iter_mut() {
            c.component_div_assign(&self.jacobian.column(0));
            c.component_div_assign(&weights);
        }

        let mut shape_func_deriv_weight = self.shape_func_deriv.clone();
        for mut c in shape_func_deriv_weight.column_iter_mut() {
            c.component_div_assign(&weights);
        }

        // Calculate residual matrix
        // let R: Matrix6xX<f64> =
        //     &qp_Fc * &shape_func_deriv_weight + &qp_Fd * &shape_func_interp_jac_weight;

        let mut R: Matrix6xX<f64> = Matrix6xX::zeros(self.nodes.nn);
        for (i, mut c) in R.column_iter_mut().enumerate() {
            let phi_i = self.shape_func_interp.column(i);
            let phip_i = self.shape_func_deriv.column(i);
            for (sl, wl) in self.qps.iter().map(|qp| qp.weight).enumerate() {
                let J = self.jacobian[(sl, 0)];
                c.add_assign((phip_i[sl] * qps[sl].Fc + J * phi_i[sl] * qps[sl].Fd) * wl)
            }
        }

        ElementDisp { qps }
    }
    fn interp_node_rotation_to_qp(&self, Rs: &[UnitQuaternion<f64>]) -> Vec<UnitQuaternion<f64>> {
        let R = MatrixXx4::from_row_iterator(
            Rs.len(),
            Rs.iter().flat_map(|&R| vec![R.w, R.i, R.j, R.k]),
        );
        let Rqp = &self.shape_func_interp * &R;
        Rqp.row_iter()
            .map(|r| UnitQuaternion::from_quaternion(Quaternion::new(r[0], r[1], r[2], r[3])))
            .collect()
    }
    fn interp_node_rotation_deriv_to_qp(&self, Rs: &[UnitQuaternion<f64>]) -> Vec<Quaternion<f64>> {
        let R = MatrixXx4::from_row_iterator(
            Rs.len(),
            Rs.iter().flat_map(|&R| vec![R.w, R.i, R.j, R.k]),
        );
        let mut Rqp_prime = &self.shape_func_deriv * &R;
        for mut c in Rqp_prime.column_iter_mut() {
            c.component_div_assign(&self.jacobian.column(0));
        }
        Rqp_prime
            .row_iter()
            .map(|r| Quaternion::new(r[0], r[1], r[2], r[3]))
            .collect()
    }
}

//------------------------------------------------------------------------------
// Element Displaced
//------------------------------------------------------------------------------

struct ElementDisp {
    qps: Vec<QuadraturePointDisp>,
}

impl ElementDisp {}

// QuadraturePointDisp defines the quadrature point data in the displaced/deformed configuration
pub struct QuadraturePointDisp {
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

impl QuadraturePointRef {
    fn sectional_strain_inertial_basis(
        &self,
        u_prime: &Vector3<f64>,
        R: &UnitQuaternion<f64>,
        R_prime: &Quaternion<f64>,
    ) -> Vector6<f64> {
        let e1 = self.x0_prime - u_prime - R * self.x0_prime;
        let e2 = kappa(R, R_prime);
        Vector6::new(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2])
    }
}

fn sectional_matrix_to_inertial_basis(RR0: &Rotation3<f64>, Astar: &Matrix6<f64>) -> Matrix6<f64> {
    let mut _RR0 = Matrix6::from_element(0.);
    _RR0.index_mut((..3, ..3)).copy_from(RR0.matrix());
    _RR0.index_mut((3.., 3..)).copy_from(RR0.matrix());
    _RR0 * Astar * _RR0.transpose()
}

fn kappa(q: &UnitQuaternion<f64>, q_prime: &Quaternion<f64>) -> Vector3<f64> {
    2. * q.F() * Vector4::new(q_prime.w, q_prime.i, q_prime.j, q_prime.k)
}

fn omega(q: &UnitQuaternion<f64>, q_dot: &Quaternion<f64>) -> Vector3<f64> {
    2. * q.F() * Vector4::new(q_dot.w, q_dot.i, q_dot.j, q_dot.k)
}

fn axial(R: &Rotation3<f64>) -> Vector3<f64> {
    Vector3::new(
        R[(2, 1)] - R[(1, 2)],
        R[(0, 2)] - R[(2, 0)],
        R[(1, 0)] - R[(0, 1)],
    ) / 2.
}

#[cfg(test)]
mod tests {

    use nalgebra::{dvector, RowVector3};

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
        let p: DVector<f64> = DVector::from_vec(gauss_legendre_lobotto_points(10));
        let s: DVector<f64> = p.add_scalar(1.) / 2.;

        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_fn(|i, j| (i * j) as f64),
        };

        // Initialize beam structure
        let beam = Beam::new(
            &s.iter()
                .map(|&si| Point4::new(fx(si), fy(si), fz(si), ft(si)))
                .collect::<Vec<Point4<f64>>>(),
            &vec![Section::new(0.0, &mat), Section::new(6.0, &mat)],
        );

        // Create element from beam
        let mut elem = beam.element(4, 7, 0., beam.length);

        // Get xyz displacements at node locations
        let u: MatrixXx3<f64> = MatrixXx3::from_rows(
            elem.nodes
                .xi
                .iter()
                .map(|&xi| {
                    let si = (xi + 1.) / 2.;
                    RowVector3::new(ux(si), uy(si), uz(si))
                })
                .collect::<Vec<RowVector3<f64>>>()
                .as_slice(),
        );

        // Get vector of unit quaternions describing the nodal rotation
        let R: Vec<UnitQuaternion<f64>> = elem
            .nodes
            .xi
            .iter()
            .map(|&xi| {
                let si = (xi + 1.) / 2.;
                UnitQuaternion::from_matrix(&rot(si))
            })
            .collect();

        // Get the deformed element
        let elem_disp = elem.displace(&u, &R);
    }
}

#[derive(Clone)]
struct Section {
    loc: f64,           // Distance along centerline from first point
    material: Material, // Material specification
}

impl Section {
    fn new(location: f64, material: &Material) -> Self {
        Section {
            loc: location,
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
