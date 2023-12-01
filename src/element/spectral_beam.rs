#![allow(non_snake_case)]

use itertools::izip;
use nalgebra::{
    distance, DMatrix, DVector, Matrix3, Matrix3x4, Matrix3xX, Matrix6, MatrixXx1, MatrixXx3,
    MatrixXx4, Point4, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4, Vector6,
};
use std::{f64::consts::PI, num::NonZeroI128};

use super::interp::{
    gauss_legendre_lobotto_points, lagrange_polynomial, lagrange_polynomial_derivative,
    quaternion_from_tangent_twist,
};

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

    fn element(&self, order: usize, loc1: f64, loc2: f64) -> Element {
        let node_xi = DVector::from(gauss_legendre_lobotto_points(order));
        let node_locs = node_xi
            .add_scalar(1.)
            .scale((loc2 - loc1) / 2.)
            .add_scalar(loc1);

        let gl_rule = gauss_quad::GaussLegendre::init(order);
        let qp_xi = DVector::from_iterator(order, gl_rule.nodes.into_iter().rev());
        let qp_weights = DVector::from_iterator(order, gl_rule.weights.into_iter().rev());
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
                        QuadraturePoint {
                            loc,
                            xi,
                            weight: weight,
                            x0: x0.transpose(),
                            x0_prime: x0_prime.transpose(),
                            R0,
                            Mstar,
                            Cstar,
                            ..Default::default()
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

#[derive(Debug)]
struct Element {
    nodes: Nodes,
    qps: Vec<QuadraturePoint>,
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

#[derive(Debug, Default)]
pub struct QuadraturePoint {
    loc: f64,
    xi: f64,
    weight: f64,
    x0: Vector3<f64>,
    x0_prime: Vector3<f64>,
    R0: UnitQuaternion<f64>,
    Mstar: Matrix6<f64>,
    Cstar: Matrix6<f64>,
    //
    u_prime: Vector3<f64>,
    R: UnitQuaternion<f64>,
    R_prime: UnitQuaternion<f64>,
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
    fn sectional_strain_inertial_basis(
        &self,
        u_prime: &Vector3<f64>,
        R: &UnitQuaternion<f64>,
        R_prime: &UnitQuaternion<f64>,
    ) -> Vector6<f64> {
        let e1 = self.x0_prime - u_prime - R * self.x0_prime;
        let e2 = kappa(R, R_prime);
        Vector6::new(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2])
    }
}

impl Element {
    fn update(&mut self, u_node: &MatrixXx3<f64>, R_node: &[UnitQuaternion<f64>]) {
        // Calculate derivative of displacement at each quadrature point
        let u_prime = (&self.shape_func_deriv * u_node).component_div(&self.jacobian);

        // Calculate displacement rotation quaternions at quadrature points
        let R = self.interp_node_rotation_to_qp(&R_node);

        // Calculate displacement rotation quaternions at quadrature points
        let R_prime = self.interp_node_rotation_deriv_to_qp(&R_node);

        // Calculate section data in inertial basis at each quadrature point
        for (qp, &R, &R_p, u_p) in izip!(
            self.qps.iter_mut(),
            R.iter(),
            R_prime.iter(),
            u_prime.transpose().column_iter()
        ) {
            let RR0 = (R * qp.R0).to_rotation_matrix();
            let mut _RR0 = Matrix6::from_element(0.);
            _RR0.index_mut((..3, ..3)).copy_from(RR0.matrix());
            _RR0.index_mut((3.., 3..)).copy_from(RR0.matrix());
            qp.u_prime = Vector3::from(u_p);
            qp.strain = qp.sectional_strain_inertial_basis(&qp.u_prime, &R, &R_p);
            qp.C = _RR0 * qp.Cstar * _RR0.transpose();
            let C11 = qp.C.index((..3, ..3));
            let C21 = qp.C.index((3.., ..3));
            let C12 = qp.C.index((..3, 3..));
            qp.M = _RR0 * qp.Mstar * _RR0.transpose();
            qp.Fc = qp.C * qp.strain;
            let Nt = qp.Fc.xyz().skew_symmetric_matrix();
            let Mt = Vector3::new(qp.Fc[3], qp.Fc[4], qp.Fc[5]).skew_symmetric_matrix();
            let x0pu0pSS: Matrix3<f64> =
                &qp.x0_prime.skew_symmetric_matrix() + &qp.u_prime.skew_symmetric_matrix();

            qp.Fd.fill(0.);
            qp.Fd
                .index_mut((3.., 0))
                .copy_from(&(x0pu0pSS.transpose() * qp.Fc.xyz()));

            qp.O.fill(0.);
            qp.O.index_mut((..3, 3..))
                .copy_from(&(-Nt + C11 * x0pu0pSS));
            qp.O.index_mut((3.., 3..))
                .copy_from(&(-Mt + C21 * x0pu0pSS));

            qp.P.fill(0.);
            qp.P.index_mut((3.., ..3))
                .copy_from(&(Nt + x0pu0pSS.transpose() * C11));
            qp.P.index_mut((3.., 3..))
                .copy_from(&(x0pu0pSS.transpose() * C12));

            qp.Q.fill(0.);
            qp.Q.index_mut((3.., 3..))
                .copy_from(&(x0pu0pSS.transpose() * (-Nt + C11 * x0pu0pSS)));
        }
    }
    fn interp_node_rotation_to_qp(&self, Rs: &[UnitQuaternion<f64>]) -> Vec<UnitQuaternion<f64>> {
        let R = MatrixXx4::from_row_iterator(
            Rs.len(),
            Rs.iter().flat_map(|&R| vec![R.i, R.j, R.k, R.w]),
        );
        println!("\ninterp node rotation");
        let Rqp = &self.shape_func_interp * &R;
        Rqp.row_iter()
            .map(|r| {
                let q = Quaternion::new(r[3], r[0], r[1], r[2]);
                let qu = UnitQuaternion::from_quaternion(q);
                let v = r.transpose().xyz();
                let w = (1. - v.dot(&v)).sqrt();
                let qua = UnitQuaternion::from_quaternion(Quaternion::new(w, v.x, v.y, v.z));
                println!("q  ={}", q);
                println!("qu ={:?}", qu.euler_angles());
                println!("qua={:?}", qua.euler_angles());
                qu
            })
            .collect()
    }
    fn interp_node_rotation_deriv_to_qp(
        &self,
        Rs: &[UnitQuaternion<f64>],
    ) -> Vec<UnitQuaternion<f64>> {
        println!("\ninterp node rotation derivative");
        let R = MatrixXx4::from_row_iterator(
            Rs.len(),
            Rs.iter().flat_map(|&R| vec![R.i, R.j, R.k, R.w]),
        );
        let mut Rqp_prime = &self.shape_func_deriv * &R;
        for mut c in Rqp_prime.column_iter_mut() {
            c.component_div_assign(&self.jacobian.column(0));
        }
        Rqp_prime
            .row_iter()
            .map(|r| {
                let q = Quaternion::new(r[3], r[0], r[1], r[2]);
                let qu = UnitQuaternion::from_quaternion(q);
                let v = r.transpose().xyz();
                let w = (1. - v.dot(&v)).sqrt();
                let qua = UnitQuaternion::from_quaternion(Quaternion::new(w, v.x, v.y, v.z));
                println!("q  ={}", q);
                println!("qu ={:?}", qu.euler_angles());
                println!("qua={:?}", qua.euler_angles());
                qu
            })
            .collect()
    }
}

fn sectional_matrix_to_inertial_basis(RR0: &Rotation3<f64>, Astar: &Matrix6<f64>) -> Matrix6<f64> {
    let mut _RR0 = Matrix6::from_element(0.);
    _RR0.index_mut((..3, ..3)).copy_from(RR0.matrix());
    _RR0.index_mut((3.., 3..)).copy_from(RR0.matrix());
    _RR0 * Astar * _RR0.transpose()
}

fn kappa(q: &UnitQuaternion<f64>, q_prime: &UnitQuaternion<f64>) -> Vector3<f64> {
    2. * q.F() * Vector4::new(q_prime.w, q_prime.i, q_prime.j, q_prime.k)
}

fn omega(q: &UnitQuaternion<f64>, q_dot: &UnitQuaternion<f64>) -> Vector3<f64> {
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
    use nalgebra::dvector;

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
        let rot = |t: f64| -> UnitQuaternion<f64> {
            UnitQuaternion::from_matrix(&Matrix3::new(
                1.,
                0.,
                0.,
                0.,
                (scale * t).cos(),
                -(scale * t).sin(),
                0.,
                (scale * t).sin(),
                (scale * t).cos(),
            ))
        };

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let p = gauss_legendre_lobotto_points(7);
        let s = DVector::from_vec(p).add_scalar(1.) / 2.;

        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_vec(vec![
                1368.17, 0.0, 0.0, 0.0, 0.0, 0.0, // Column 1
                0.0, 88.56, 0.0, 0.0, 0.0, 0.0, // Column 2
                0.0, 0.0, 38.78, 0.0, 0.0, 0.0, // Column 3
                0.0, 0.0, 0.0, 16.96, 17.61, -0.351, // Column 4
                0.0, 0.0, 0.0, 17.61, 59.12, -0.370, // Column 5
                0.0, 0.0, 0.0, -0.351, -0.370, 141.47, // Column 6
            ]),
        };

        let beam = Beam::new(
            &s.iter()
                .map(|&si| Point4::new(fx(si), fy(si), fz(si), ft(si)))
                .collect::<Vec<Point4<f64>>>(),
            &vec![Section::new(0.0, &mat), Section::new(10.0, &mat)],
        );

        let mut elem = beam.element(7, 0., beam.length);

        // Get xyz displacements at node locations
        let u = MatrixXx3::from_row_iterator(
            elem.nodes.locs.len(),
            elem.nodes
                .locs
                .iter()
                .flat_map(|&si| vec![ux(si), uy(si), uz(si)]),
        );

        // Get vector of unit quaternions describing the nodal rotation
        let R: Vec<UnitQuaternion<f64>> = elem.nodes.locs.iter().map(|&si| rot(si)).collect();

        elem.update(&u, &R);
    }

    #[test]
    fn create_beam() {
        // Create material
        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_vec(vec![
                1368.17, 0.0, 0.0, 0.0, 0.0, 0.0, // Column 1
                0.0, 88.56, 0.0, 0.0, 0.0, 0.0, // Column 2
                0.0, 0.0, 38.78, 0.0, 0.0, 0.0, // Column 3
                0.0, 0.0, 0.0, 16.96, 17.61, -0.351, // Column 4
                0.0, 0.0, 0.0, 17.61, 59.12, -0.370, // Column 5
                0.0, 0.0, 0.0, -0.351, -0.370, 141.47, // Column 6
            ]),
        };

        // Check material creation
        assert_eq!(mat.stiffness.m54, 17.61);
        assert_eq!(mat.mass.m11, 1.0);
        assert_eq!(mat.mass.m12, 0.0);
        assert_eq!(mat.eta.magnitude(), 0.0);

        // Create vector of sections
        let sections = vec![Section::new(0.0, &mat), Section::new(10.0, &mat)];

        // Create vector of beam reference line points
        let points = vec![
            Point4::new(0.0, 0.0, 0.0, 0.0),
            Point4::new(10.0, 0.0, 0.0, 0.0),
        ];

        // Create beam
        let beam = Beam::new(&points, &sections);

        // Check beam creation
        assert_eq!(beam.length, 10.0);
        assert_eq!(beam.ref_locs, dvector![0.0, 10.0]);

        // Create element
        let elem = beam.element(5, 0., 10.);
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
