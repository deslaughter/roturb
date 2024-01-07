#![allow(non_snake_case)]
#![allow(dead_code)]

use super::{
    interp::{lagrange_polynomial, lagrange_polynomial_derivative},
    quadrature::Quadrature,
};
use crate::prelude::*;
use serde::Serialize;

//------------------------------------------------------------------------------
// Element
//------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct Element {
    pub q_root: Vector7,
    pub nodes: Nodes,
    pub qps: Vec<QuadraturePoint>,
    pub shape_func_interp: MatrixNxQ,
    pub shape_func_deriv: MatrixNxQ,
    pub jacobian: Matrix1xX,
}

impl Element {
    pub fn update_states(&mut self, Q: &Matrix7xX, V: &Matrix6xX, A: &Matrix6xX, g: &Vector3) {
        // Save node displacement and rotation
        self.nodes.u.copy_from(&Q.fixed_rows::<3>(0));
        self.nodes.r.copy_from(&Q.fixed_rows::<4>(3));

        // Save node velocities
        self.nodes.u_dot.copy_from(&V.fixed_rows::<3>(0));
        self.nodes.omega.copy_from(&V.fixed_rows::<3>(3));

        // Save node accelerations
        self.nodes.u_ddot.copy_from(&A.fixed_rows::<3>(0));
        self.nodes.omega_dot.copy_from(&A.fixed_rows::<3>(3));

        // Interpolate displacement and displacement derivative to quadrature points
        let u: Matrix3xX = interp_position(&self.shape_func_interp, &self.nodes.u);
        let u_prime: Matrix3xX =
            interp_position_derivative(&self.shape_func_deriv, &self.jacobian, &self.nodes.u);

        // Interpolate rotation and rotation derivative to quadrature points
        let r: Matrix4xX = interp_rotation(&self.shape_func_interp, &self.nodes.r);
        let r_prime: Matrix4xX =
            interp_rotation_derivative(&self.shape_func_deriv, &self.jacobian, &self.nodes.r);

        // Interpolate velocities to quadrature points
        let u_dot: Matrix3xX = interp_position(&self.shape_func_interp, &self.nodes.u_dot);
        let omega: Matrix3xX = interp_position(&self.shape_func_interp, &self.nodes.omega);

        // Interpolate accelerations to quadrature points
        let u_ddot: Matrix3xX = interp_position(&self.shape_func_interp, &self.nodes.u_ddot);
        let omega_dot: Matrix3xX = interp_position(&self.shape_func_interp, &self.nodes.omega_dot);

        // Calculate quadrature point values with applied displacement/rotation
        for (i, qp) in self.qps.iter_mut().enumerate() {
            qp.update_states(
                &Vector3::from(u.column(i)),
                &Vector3::from(u_prime.column(i)),
                &Vector3::from(u_dot.column(i)),
                &Vector3::from(u_ddot.column(i)),
                &Vector4::from(r.column(i)).as_unit_quaternion(),
                &Vector4::from(r_prime.column(i)).as_quaternion(),
                &Vector3::from(omega.column(i)),
                &Vector3::from(omega_dot.column(i)),
                g,
            )
        }
    }

    pub fn apply_force(&mut self, forces: &Matrix6xX) {
        self.nodes.F.copy_from(forces);
    }

    /// Spectral finite-element part of iteration matrix
    pub fn K_FE(&self) -> MatrixD {
        self.K_I() + self.K_E()
    }

    pub fn K_E(&self) -> MatrixN {
        let mut K_E: MatrixN = MatrixN::zeros(6 * self.nodes.num, 6 * self.nodes.num);
        for i in 0..self.nodes.num {
            for j in 0..self.nodes.num {
                let mut Kij = K_E.fixed_view_mut::<6, 6>(i * 6, j * 6);
                for (sl, qp) in self.qps.iter().enumerate() {
                    let phi_i = self.shape_func_interp[(i, sl)];
                    let phi_j = self.shape_func_interp[(j, sl)];
                    let phi_prime_i = self.shape_func_deriv[(i, sl)];
                    let phi_prime_j = self.shape_func_deriv[(j, sl)];
                    let J = self.jacobian[sl];
                    Kij += qp.weight
                        * (phi_i * qp.Puu * phi_prime_j
                            + phi_i * qp.Quu * phi_j * J
                            + phi_prime_i * qp.Cuu * phi_prime_j / J
                            + phi_prime_i * qp.Ouu * phi_j);
                }
            }
        }
        K_E
    }

    pub fn M(&self) -> MatrixD {
        self.integrate_matrices(self.qps.iter().map(|qp| qp.Muu).collect_vec().as_slice())
    }

    pub fn G(&self) -> MatrixD {
        self.integrate_matrices(self.qps.iter().map(|qp| qp.Guu).collect_vec().as_slice())
    }

    pub fn K_I(&self) -> MatrixD {
        self.integrate_matrices(self.qps.iter().map(|qp| qp.Kuu).collect_vec().as_slice())
    }

    fn integrate_matrices(&self, mats: &[Matrix6]) -> MatrixD {
        let mut M: MatrixD = MatrixD::zeros(6 * self.nodes.num, 6 * self.nodes.num);
        for i in 0..self.nodes.num {
            for j in 0..self.nodes.num {
                let mut Mij = M.fixed_view_mut::<6, 6>(i * 6, j * 6);
                for (sl, qp) in self.qps.iter().enumerate() {
                    let phi_i = self.shape_func_interp[(i, sl)];
                    let phi_j = self.shape_func_interp[(j, sl)];
                    Mij += qp.weight * phi_i * mats[sl] * phi_j * self.jacobian[sl];
                }
            }
        }
        M
    }

    /// Nodal residual force vector
    pub fn R_FE(&self) -> VectorD {
        let F_E = self.F_E();
        let F_I = self.F_I();
        let F_ext = self.F_ext();
        let F_g = self.F_g();
        F_I + F_E - F_ext - F_g
    }

    /// Elastic nodal force vector
    pub fn F_E(&self) -> VectorD {
        let mut FE: Matrix6xX = Matrix6xX::zeros(self.nodes.num);
        for (i, mut col) in FE.column_iter_mut().enumerate() {
            for (sl, qp) in self.qps.iter().enumerate() {
                col.add_assign(
                    qp.weight
                        * (self.shape_func_deriv[(i, sl)] * qp.F_C
                            + self.jacobian[sl] * self.shape_func_interp[(i, sl)] * qp.F_D),
                );
            }
        }
        VectorD::from_column_slice(FE.as_slice())
    }

    /// Gravity nodal force vector
    pub fn F_g(&self) -> VectorD {
        self.integrate_vectors(&self.qps.iter().map(|qp| qp.F_G).collect_vec())
    }

    /// Inertial nodal force vector
    pub fn F_I(&self) -> VectorD {
        self.integrate_vectors(&self.qps.iter().map(|qp| qp.F_I).collect_vec())
    }

    /// External nodal force vector
    pub fn F_ext(&self) -> VectorD {
        self.integrate_vectors(&self.qps.iter().map(|qp| qp.F_ext).collect_vec())
            + VectorD::from_column_slice(self.nodes.F.as_slice())
    }

    fn integrate_vectors(&self, vecs: &[Vector6]) -> VectorD {
        let mut V: Matrix6xX = Matrix6xX::zeros(self.nodes.num);
        for (i, mut col) in V.column_iter_mut().enumerate() {
            for (sl, qp) in self.qps.iter().enumerate() {
                col.add_assign(
                    self.shape_func_interp[(i, sl)] * vecs[sl] * self.jacobian[sl] * qp.weight,
                );
            }
        }
        VectorD::from_column_slice(V.as_slice())
    }

    pub fn constraint_residual_vector(&self) -> Vector6 {
        // Root rotation as unit quaternions
        let root_r = self
            .q_root
            .fixed_rows::<4>(3)
            .clone_owned()
            .as_unit_quaternion()
            .scaled_axis();
        // Node 1 rotation as unit quaternions
        let n1_r = self
            .nodes
            .r
            .fixed_columns::<1>(0)
            .clone_owned()
            .as_unit_quaternion()
            .scaled_axis();
        // let diff_r = (root_r.inverse() * n1_r).scaled_axis();
        Vector6::new(
            self.nodes.u[0] - self.q_root[0],
            self.nodes.u[1] - self.q_root[1],
            self.nodes.u[2] - self.q_root[2],
            n1_r[0] - root_r[0],
            n1_r[1] - root_r[1],
            n1_r[2] - root_r[2],
            // diff_r[0],
            // diff_r[1],
            // diff_r[2],
        )
    }

    pub fn constraints_gradient_matrix(&self) -> Matrix6xX {
        let mut B = Matrix6xX::zeros(self.nodes.num * 6);
        B.fill_diagonal(1.);
        B
    }
}

//------------------------------------------------------------------------------
// Nodes
//------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Nodes {
    pub num: usize,
    pub s: VectorN,
    pub xi: VectorN,
    pub x0: Matrix3xX,
    pub r0: Matrix4xX,
    pub u: Matrix3xX,
    pub u_dot: Matrix3xX,
    pub u_ddot: Matrix3xX,
    pub r: Matrix4xX,
    pub omega: Matrix3xX,
    pub omega_dot: Matrix3xX,
    pub F: Matrix6xX,
}

impl Nodes {
    pub fn new(s: &VectorN, xi: &VectorN, x0: &Matrix3xX, r0: &Matrix4xX) -> Self {
        let num_nodes = s.len();
        Nodes {
            num: num_nodes,
            s: s.clone(),
            xi: xi.clone(),
            x0: x0.clone(),
            r0: r0.clone(),
            u: Matrix3xX::zeros(num_nodes),
            u_dot: Matrix3xX::zeros(num_nodes),
            u_ddot: Matrix3xX::zeros(num_nodes),
            r: Matrix4xX::zeros(num_nodes),
            omega: Matrix3xX::zeros(num_nodes),
            omega_dot: Matrix3xX::zeros(num_nodes),
            F: Matrix6xX::zeros(num_nodes),
        }
    }
    pub fn element(&self, quadrature: &Quadrature, sections: &[Section]) -> Element {
        let qp_xi: VectorQ = quadrature.points.clone();
        let qp_weights: VectorQ = quadrature.weights.clone();
        let num_qp = qp_xi.len();

        // Calculate shape functions weights to interpolate from nodes to quadrature points [nq x nn]
        let shape_func_interp: MatrixNxQ = MatrixNxQ::from_iterator(
            self.num,
            num_qp,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial(xi, self.xi.as_slice())),
        );

        // Calculate shape functions weights to take derivative from nodes to quadrature points [nq x nn]
        let shape_func_deriv: MatrixNxQ = MatrixNxQ::from_iterator(
            self.num,
            num_qp,
            qp_xi
                .iter()
                .flat_map(|&xi| lagrange_polynomial_derivative(xi, self.xi.as_slice())),
        );

        // Get section locations as xi
        let section_xi = sections
            .iter()
            .map(|section| (section.s + 1.) / 2.)
            .collect_vec();

        // Calculate the interpolated mass matrix at each quadrature point
        let qp_mass: Vec<Matrix6> = qp_xi
            .iter()
            .map(|&xi| {
                let weights = lagrange_polynomial(xi, &section_xi);
                sections
                    .iter()
                    .zip(weights.iter())
                    .map(|(section, &w)| section.material.M_star * w)
                    .sum()
            })
            .collect();

        // Calculate the interpolated stiffness matrix at each quadrature point
        let qp_stiffness: Vec<Matrix6> = qp_xi
            .iter()
            .map(|&xi| {
                let weights = lagrange_polynomial(xi, &section_xi);
                sections
                    .iter()
                    .zip(weights.iter())
                    .map(|(section, &w)| section.material.C_star * w)
                    .sum()
            })
            .collect();

        // Get derivative of node positions wrt xi at the quadrature points
        let qp_deriv: Matrix3xX = &self.x0 * &shape_func_deriv;

        // Calculate Jacobian matrix
        let jacobian: Matrix1xX =
            Matrix1xX::from_iterator(qp_xi.len(), qp_deriv.column_iter().map(|r| r.norm()));

        // Interpolate the position, rotation from nodes to the quadrature points
        let qp_x0: Matrix3xX = interp_position(&shape_func_interp, &self.x0);
        let qp_r0: Matrix4xX = interp_rotation(&shape_func_interp, &self.r0);

        // Convert rotation matrix to unit quaternions
        let qp_R0: Vec<UnitQuaternion> = qp_r0
            .column_iter()
            .map(|r| Vector4::from(r).as_unit_quaternion())
            .collect();

        // Calculate the tangent vector of the position at each quadrature point
        let qp_x0_prime: Matrix3xX =
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
                F_ext: Vector6::zeros(),
                ..Default::default()
            },
        )
        .collect();

        // Build and return element
        Element {
            q_root: Vector7::from_vec(vec![0., 0., 0., 1., 0., 0., 0.]),
            nodes: self.clone(),
            qps,
            shape_func_interp,
            shape_func_deriv,
            jacobian,
        }
    }
}

//------------------------------------------------------------------------------
// Quadrature Points
//------------------------------------------------------------------------------

// QuadraturePoint defines the quadrature point data in the reference configuration
#[derive(Debug, Default, Serialize)]
pub struct QuadraturePoint {
    xi: f64,
    x0: Vector3,
    x0_prime: Vector3,
    R0: UnitQuaternion,
    M_star: Matrix6,
    C_star: Matrix6,
    // quadrature integration weight
    weight: f64,

    /// translational displacement
    u: Vector3,
    /// translational displacement derivative
    u_prime: Vector3,
    /// translational velocity
    u_dot: Vector3,
    /// translational acceleration
    u_ddot: Vector3,
    /// angular displacement
    R: UnitQuaternion,
    /// angular displacement derivative
    R_prime: Quaternion,
    /// angular velocity
    omega: Vector3,
    omega_dot: Vector3,
    RR0: Rotation3,
    /// Strain vector
    strain: Vector6,
    /// Mass matrix
    Muu: Matrix6,
    /// Stiffness matrix
    Cuu: Matrix6,
    /// Elastic forces
    F_C: Vector6,
    /// Elastic forces
    F_D: Vector6,
    /// Inertial forces
    F_I: Vector6,
    /// External forces
    F_ext: Vector6,
    /// Gravity forces
    F_G: Vector6,
    Ouu: Matrix6,
    Puu: Matrix6,
    Quu: Matrix6,
    Guu: Matrix6,
    Kuu: Matrix6,
}

impl QuadraturePoint {
    pub fn update_states(
        &mut self,
        u: &Vector3,
        u_prime: &Vector3,
        u_dot: &Vector3,
        u_ddot: &Vector3,
        R: &UnitQuaternion,
        R_prime: &Quaternion,
        omega: &Vector3,
        omega_dot: &Vector3,
        gravity: &Vector3,
    ) {
        // Update displacement inputs
        self.u = *u;
        self.u_prime = *u_prime;
        self.u_dot = *u_dot;
        self.u_ddot = *u_ddot;
        self.R = *R;
        self.R_prime = *R_prime;
        self.omega = *omega;
        self.omega_dot = *omega_dot;

        // Calculate 6x6 rotation matrix
        self.RR0 = (R * self.R0).to_rotation_matrix();
        let mut RR0e: Matrix6 = Matrix6::zeros();
        RR0e.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.RR0.matrix());
        RR0e.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(self.RR0.matrix());

        // Sectional mass matrix
        self.Muu = RR0e * self.M_star * RR0e.transpose();

        // Components of section mass matrix
        let m = self.Muu[(0, 0)];
        let eta: Vector3 = if m == 0. {
            Vector3::zeros()
        } else {
            Vector3::new(
                self.Muu[(5, 1)] / m,
                -self.Muu[(5, 0)] / m,
                self.Muu[(4, 0)] / m,
            )
        };
        let rho: Matrix3 = Matrix3::from(self.Muu.fixed_view::<3, 3>(3, 3));

        // Sectional stiffness matrix
        self.Cuu = RR0e * self.C_star * RR0e.transpose();

        // Components of section stiffnes matrix
        let C11 = self.Cuu.fixed_view::<3, 3>(0, 0);
        let C21 = self.Cuu.fixed_view::<3, 3>(3, 0);
        let C12 = self.Cuu.fixed_view::<3, 3>(0, 3);

        // Sectional strain
        let e1: Vector3 = self.x0_prime + u_prime - R * self.x0_prime;
        let e2: Vector3 = 2. * R.E() * R_prime.wijk();
        // let e1: Vector3 = self.x0_prime + u_prime - self.RR0.matrix().column(0);
        // let R_prime_Rt = R.to_rotation_matrix()
        //     * (2. * R.G() * R_prime.G().transpose())
        //     * R.to_rotation_matrix().transpose();
        // let e2 = Vector3::new(
        //     R_prime_Rt[(2, 1)] - R_prime_Rt[(1, 2)],
        //     R_prime_Rt[(0, 2)] - R_prime_Rt[(2, 0)],
        //     R_prime_Rt[(1, 0)] - R_prime_Rt[(0, 1)],
        // ) / 2.;
        self.strain = Vector6::new(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]);

        // Calculate skew_symmetric_matrix(x0_prime + u_prime)
        let x0pupSS: Matrix3 = self.x0_prime.tilde() + u_prime.tilde();

        // Elastic force C
        self.F_C = self.Cuu * self.strain;
        let N: Vector3 = Vector3::new(self.F_C[0], self.F_C[1], self.F_C[2]);
        let M: Vector3 = Vector3::new(self.F_C[3], self.F_C[4], self.F_C[5]);

        // Elastic force D
        self.F_D.fill(0.);
        self.F_D
            .fixed_rows_mut::<3>(3)
            .copy_from(&(x0pupSS.transpose() * N));

        // Inertial force
        self.F_I.fixed_rows_mut::<3>(0).copy_from(
            &(m * u_ddot + (omega_dot.tilde() + omega.tilde() * omega.tilde()) * m * eta),
        );
        self.F_I
            .fixed_rows_mut::<3>(3)
            .copy_from(&(m * eta.tilde() * u_ddot + rho * omega_dot + omega.tilde() * rho * omega));

        // Gravity force
        self.F_G.fixed_rows_mut::<3>(0).copy_from(&(m * gravity));
        self.F_G
            .fixed_rows_mut::<3>(3)
            .copy_from(&(m * eta.tilde() * gravity));

        // Linearization matrices (stiffness)
        self.Ouu.fill(0.);
        self.Ouu
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-N.tilde() + C11 * x0pupSS));
        self.Ouu
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-M.tilde() + C21 * x0pupSS));

        self.Puu.fill(0.);
        self.Puu
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(N.tilde() + x0pupSS.transpose() * C11));
        self.Puu
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * C12));

        self.Quu.fill(0.);
        self.Quu
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(x0pupSS.transpose() * (-N.tilde() + C11 * x0pupSS)));

        // Inertia gyroscopic matrix
        self.Guu.fill(0.);
        self.Guu.fixed_view_mut::<3, 3>(0, 3).copy_from(
            &((omega.tilde() * m * eta).tilde().transpose()
                + omega.tilde() * m * eta.tilde().transpose()),
        );
        self.Guu
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(omega.tilde() * rho - (rho * omega).tilde()));

        // Inertia stiffness matrix
        self.Kuu.fill(0.);
        self.Kuu.fixed_view_mut::<3, 3>(0, 3).copy_from(
            &((omega_dot.tilde() + omega.tilde() * omega.tilde()) * m * eta.tilde().transpose()),
        );
        self.Kuu.fixed_view_mut::<3, 3>(3, 3).copy_from(
            &(u_ddot.tilde() * m * eta.tilde()
                + (rho * omega_dot.tilde() - (rho * omega_dot).tilde())
                + omega.tilde() * (rho * omega.tilde() - (rho * omega).tilde())),
        );
    }
}

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    use approx::assert_relative_eq;

    use crate::element::interp::{gauss_legendre_lobotto_points, quaternion_from_tangent_twist};

    #[test]
    pub fn test_beam_formulation() {
        let fz = |t: f64| -> f64 { t - 2. * t * t };
        let fy = |t: f64| -> f64 { -2. * t + 3. * t * t };
        let fx = |t: f64| -> f64 { 5. * t };
        // let ft = |t: f64| -> f64 { 0. * t * t };

        let scale = 0.1;
        let ux = |t: f64| -> f64 { scale * t * t };
        let uy = |t: f64| -> f64 { scale * (t * t * t - t * t) };
        let uz = |t: f64| -> f64 { scale * (t * t + 0.2 * t * t * t) };
        let rot = |t: f64| -> Matrix3 {
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

        // Velocities
        let vx = |s: f64| -> f64 { scale * (s) };
        let vy = |s: f64| -> f64 { scale * (s * s - s) };
        let vz = |s: f64| -> f64 { scale * (s * s + 0.2 * s * s * s) };
        let omega_x = |s: f64| -> f64 { scale * (s) };
        let omega_y = |s: f64| -> f64 { scale * (s * s - s) };
        let omega_z = |s: f64| -> f64 { scale * (s * s + 0.2 * s * s * s) };

        // Displacements
        let ax = |s: f64| -> f64 { scale * (s) };
        let ay = |s: f64| -> f64 { scale * (2. * s * s - s) };
        let az = |s: f64| -> f64 { scale * (2. * s * s + 0.2 * s * s * s) };
        let omega_dot_x = |s: f64| -> f64 { scale * (s) };
        let omega_dot_y = |s: f64| -> f64 { scale * (s * s - s) };
        let omega_dot_z = |s: f64| -> f64 { scale * (s * s - s) };

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(4));
        let s: VectorN = xi.add_scalar(1.) / 2.;

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

        // Construct mass matrix
        let eta_star: Vector3 = Vector3::new(0.1, 0.2, 0.3);
        let m = 2.0;
        let mut mass: Matrix6 = Matrix6::zeros();
        mass.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::from_diagonal_element(m));
        mass.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(m * eta_star.tilde().transpose()));
        mass.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(m * eta_star.tilde()));
        mass.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&Matrix3::from_fn(|i, j| ((i + 1) * (j + 1)) as f64));

        // Create material
        let mat = Material {
            M_star: mass,
            C_star: Matrix6::from_fn(|i, j| ((i + 1) * (j + 1)) as f64),
        };

        assert_relative_eq!(
            mat.M_star,
            Matrix6::from_vec(vec![
                2., 0., 0., 0., 0.6, -0.4, // column 1
                0., 2., 0., -0.6, 0., 0.2, // column 2
                0., 0., 2., 0.4, -0.2, 0., // column 3
                0., -0.6, 0.4, 1., 2., 3., // column 4
                0.6, 0., -0.2, 2., 4., 6., // column 5
                -0.4, 0.2, 0., 3., 6., 9., // column 6
            ])
        );

        assert_relative_eq!(
            mat.C_star,
            Matrix6::from_vec(vec![
                1., 2., 3., 4., 5., 6., // column 1
                2., 4., 6., 8., 10., 12., // column 2
                3., 6., 9., 12., 15., 18., // column 3
                4., 8., 12., 16., 20., 24., // column 4
                5., 10., 15., 20., 25., 30., // column 5
                6., 12., 18., 24., 30., 36., // column 6
            ])
        );

        // Create quadrature points and weights
        let gq = Quadrature::gauss(7);

        // Create element from nodes
        let mut elem = nodes.element(&gq, &vec![Section::new(0.0, &mat), Section::new(1.0, &mat)]);

        // Get xyz displacements at node locations
        let u: Matrix3xX = Matrix3xX::from_columns(
            s.iter()
                .map(|&si| Vector3::new(ux(si), uy(si), uz(si)))
                .collect_vec()
                .as_slice(),
        );

        // Get vector of unit quaternions describing the nodal rotation
        let r: Matrix4xX = Matrix4xX::from_columns(
            s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).wijk())
                .collect_vec()
                .as_slice(),
        );

        // Combine translation and rotation displacements
        let mut Q: Matrix7xX = Matrix7xX::zeros(nodes.num);
        Q.fixed_rows_mut::<3>(0).copy_from(&u);
        Q.fixed_rows_mut::<4>(3).copy_from(&r);

        // Combine translation and angular velocities
        let mut V: Matrix6xX = Matrix6xX::zeros(nodes.num);
        for (mut c, &s) in V.column_iter_mut().zip(s.iter()) {
            c.copy_from(&Vector6::new(
                vx(s),
                vy(s),
                vz(s),
                omega_x(s),
                omega_y(s),
                omega_z(s),
            ))
        }
        let mut A: Matrix6xX = Matrix6xX::zeros(nodes.num);
        for (mut c, &s) in A.column_iter_mut().zip(s.iter()) {
            c.copy_from(&Vector6::new(
                ax(s),
                ay(s),
                az(s),
                omega_dot_x(s),
                omega_dot_y(s),
                omega_dot_z(s),
            ))
        }

        // Gravity
        let g = Vector3::zeros();

        // Displace the element
        elem.update_states(&Q, &V, &A, &g);

        assert_relative_eq!(
            elem.qps[0].u,
            Vector3::new(0.0000647501, -0.0000631025, 0.0000650796),
            epsilon = 1.0e-8
        );

        assert_relative_eq!(
            elem.qps[0].R.wijk(),
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
            elem.qps[0].F_C,
            Vector6::new(0.10234, 0.151237, 0.278871, 0.400353, 0.324973, 0.587574),
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            elem.qps[0].F_D,
            Vector6::new(0., 0., 0., 0.120831, 0.241111, -0.175102),
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            elem.qps[0].F_I,
            Vector6::new(
                0.00437542,
                -0.00699735,
                0.00168548,
                -0.00883081,
                -0.0137881,
                -0.0297526,
            ),
            epsilon = 1.0e-5
        );

        // Get elastic force vector
        let R_E: VectorN = elem.F_E();
        assert_relative_eq!(
            R_E,
            VectorN::from_vec(vec![
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

        // Get inertial force vector
        let R_I: VectorN = elem.F_I();
        assert_relative_eq!(
            R_I,
            VectorN::from_vec(vec![
                0.0001160455640892761,
                -0.0006507362696178125,
                -0.0006134866787567512,
                0.0006142322011934131,
                -0.002199479688149198,
                -0.002486843354672648,
                0.04224129527348784,
                -0.04970288500953023,
                0.03088887284431367,
                -0.06975512417597242,
                -0.1016119340697187,
                -0.2145759755006085,
                0.1782195482379224,
                -0.06852557905980959,
                0.239183232808296,
                -0.1174211448270913,
                -0.2577547967767749,
                -0.3851496964119589,
                0.2842963634125303,
                0.09461848004333057,
                0.5753721231363037,
                -0.0372222680242176,
                -0.08186055756075448,
                -0.02397231276703171,
                0.07251940986370667,
                0.04758219371046168,
                0.1727148583550362,
                -0.007667261048491473,
                0.02731289347020924,
                0.05581821163081066,
            ]),
            epsilon = 1.0e-7
        );

        // Get elastic stiffness matrix
        let K_E: MatrixN = elem.K_E();
        assert_relative_eq!(K_E[(0, 0)], 1.7424036187210084, epsilon = 1.0e-12);
        assert_relative_eq!(K_E[(0, 1)], 2.596502219359714, epsilon = 1.0e-12);

        // Get inertial stiffness matrix
        let K_I: MatrixN = elem.K_I();
        assert_relative_eq!(K_I[(0, 0)], 0., epsilon = 1e-12);
        assert_relative_eq!(K_I[(0, 3)], -0.00103122831912768, epsilon = 1e-12);

        // Get gyroscopic matrix
        let G: MatrixN = elem.G();
        assert_relative_eq!(G[(0, 0)], 0., epsilon = 1e-12);
        assert_relative_eq!(G[(0, 3)], -0.000142666315350446, epsilon = 1e-12);

        // Get mass matrix
        let M: MatrixN = elem.M();
        assert_relative_eq!(M[(0, 0)], 0.477242989475536, epsilon = 1e-12);
        assert_relative_eq!(M[(4, 0)], 0.148588501603872, epsilon = 1e-12);
    }

    #[test]
    fn test_displaced_beam() {
        //----------------------------------------------------------------------
        // Initial configuration
        //----------------------------------------------------------------------

        let fx = |s: f64| -> f64 { 10. * s + 1. };
        let fz = |s: f64| -> f64 { 0. * s };
        let fy = |s: f64| -> f64 { 0. * s };
        let ft = |s: f64| -> f64 { 0. * s };
        let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(4));
        let s: VectorN = xi.add_scalar(1.) / 2.;

        let num_nodes = s.len();

        // Node initial position
        let x0: Matrix3xX = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![fx(si), fy(si), fz(si)]),
        );
        assert_relative_eq!(
            x0,
            Matrix3xX::from_vec(vec![
                1.,
                0.,
                0., // column 1
                2.726731646460114,
                0.,
                0., // column 2
                6.,
                0.,
                0., // column 3
                9.273268353539887,
                0.,
                0., // column 4
                11.,
                0.,
                0., // column 5
            ]),
        );

        // Node initial rotation
        let interp_deriv: MatrixNxQ = MatrixNxQ::from_iterator(
            num_nodes,
            num_nodes,
            s.iter()
                .flat_map(|&si| lagrange_polynomial_derivative(si, s.as_slice())),
        );
        let mut tangent: Matrix3xX = &x0 * interp_deriv;
        for mut c in tangent.column_iter_mut() {
            c.normalize_mut();
        }

        let r0: Matrix4xX = Matrix4xX::from_columns(
            s.iter()
                .zip(tangent.column_iter())
                .map(|(&si, tan)| quaternion_from_tangent_twist(&Vector3::from(tan), ft(si)).wijk())
                .collect::<Vec<Vector4>>()
                .as_slice(),
        );
        assert_relative_eq!(
            r0,
            Matrix4xX::from_vec(vec![
                1., 0., 0., 0., // column 1
                1., 0., 0., 0., // column 2
                1., 0., 0., 0., // column 3
                1., 0., 0., 0., // column 4
                1., 0., 0., 0., // column 5
            ])
        );

        // Create nodes structure
        let nodes = Nodes::new(&s, &xi, &x0, &r0);

        //----------------------------------------------------------------------
        // Displacements and rotations from reference
        //----------------------------------------------------------------------

        let scale = 0.0;
        let ux = |t: f64| -> f64 { scale * t * t };
        let uy = |t: f64| -> f64 {
            if t == 1. {
                0.001
            } else {
                scale * (t * t * t - t * t)
            }
        };
        let uz = |t: f64| -> f64 { scale * (t * t + 0.2 * t * t * t) };
        let rot = |t: f64| -> Matrix3 {
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

        // Get xyz displacements at node locations
        let u: Matrix3xX = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![ux(si), uy(si), uz(si)]),
        );

        assert_relative_eq!(
            u,
            Matrix3xX::from_vec(vec![
                0., 0., 0., // column 1
                0., 0., 0., // column 2
                0., 0., 0., // column 3
                0., 0., 0., // column 4
                0., 0.001, 0., // column 5
            ]),
        );

        // Get matrix describing the nodal rotation
        let r: Matrix4xX = Matrix4xX::from_columns(
            &s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).wijk())
                .collect_vec(),
        );

        // Quadrature rule
        let gq = Quadrature::gauss(7);

        // Create material
        let mat = Material {
            M_star: Matrix6::identity(),
            C_star: Matrix6::from_row_slice(&vec![
                1.36817e6, 0., 0., 0., 0., 0., // row 1
                0., 88560., 0., 0., 0., 0., // row 2
                0., 0., 38780., 0., 0., 0., // row 3
                0., 0., 0., 16960., 17610., -351., // row 4
                0., 0., 0., 17610., 59120., -370., // row 5
                0., 0., 0., -351., -370., 141470., // row 6
            ]),
        };

        // Create sections
        let sections: Vec<Section> = vec![Section::new(0.0, &mat), Section::new(1.0, &mat)];

        // Create element from nodes
        let mut elem = nodes.element(&gq, &sections);

        // Gravity
        let g = Vector3::zeros();

        // Combine translation and rotation displacements
        let mut Q: Matrix7xX = Matrix7xX::zeros(nodes.num);
        Q.fixed_rows_mut::<3>(0).copy_from(&u);
        Q.fixed_rows_mut::<4>(3).copy_from(&r);

        // Combine translation and angular velocities
        let V: Matrix6xX = Matrix6xX::zeros(nodes.num);
        let A: Matrix6xX = Matrix6xX::zeros(nodes.num);

        // Get the deformed element
        elem.update_states(&Q, &V, &A, &g);

        // Get constraints_gradient_matrix
        let B: Matrix6xX = elem.constraints_gradient_matrix();
        assert_relative_eq!(B.columns(0, 6), Matrix6::identity().columns(0, 6));

        // Get residual vector
        let R: VectorN = elem.R_FE();
        assert_relative_eq!(
            R,
            VectorN::from_vec(vec![
                0., 0.8856, 0., 0., 0., 4.428, 0., -3.01898, 0., 0., 0., -12.4884, 0., 9.4464, 0.,
                0., 0., 23.616, 0., -69.305, 0., 0., 0., -59.8356, 0., 61.992, 0., 0., 0., -44.28
            ]),
            epsilon = 1e-4
        );

        // Get constraint_residual_vector
        let Phi: Vector6 = elem.constraint_residual_vector();
        assert_relative_eq!(Phi, Vector6::zeros());

        // Static iteration matrix
        let K_FE: MatrixN = elem.K_E();
        assert_relative_eq!(K_FE[(0, 0)], 957719.0, epsilon = 1e-5);
        assert_relative_eq!(K_FE[(1, 1)], 61992.0, epsilon = 1e-5);
    }
}

#[derive(Clone, Serialize)]
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

#[derive(Clone, Serialize)]
pub struct Material {
    pub M_star: Matrix6, // Mass matrix
    pub C_star: Matrix6, // Stiffness matrix
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

fn interp_force(shape_func_interp: &MatrixNxQ, node_force: &Matrix6xX) -> Matrix6xX {
    node_force * shape_func_interp
}

fn interp_position(shape_func_interp: &MatrixNxQ, node_position: &Matrix3xX) -> Matrix3xX {
    node_position * shape_func_interp
}

fn interp_rotation(shape_func_interp: &MatrixNxQ, node_rotation: &Matrix4xX) -> Matrix4xX {
    let mut r: Matrix4xX = node_rotation * shape_func_interp;
    for mut c in r.column_iter_mut() {
        c.normalize_mut();
    }
    r
}

fn interp_position_derivative(
    shape_func_deriv: &MatrixNxQ,
    jacobian: &Matrix1xX,
    node_position: &Matrix3xX,
) -> Matrix3xX {
    let mut u_prime: Matrix3xX = node_position * shape_func_deriv;
    for mut r in u_prime.row_iter_mut() {
        r.component_div_assign(jacobian);
    }
    u_prime
}

fn interp_rotation_derivative(
    shape_func_deriv: &MatrixNxQ,
    jacobian: &Matrix1xX,
    node_rotation: &Matrix4xX,
) -> Matrix4xX {
    let mut r_prime: Matrix4xX = node_rotation * shape_func_deriv;
    for mut r in r_prime.row_iter_mut() {
        r.component_div_assign(jacobian);
    }
    r_prime
}
