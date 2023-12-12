#![allow(non_snake_case)]

use std::ops::AddAssign;

use crate::element::{
    gebt::{Element, Nodes, QuatExt, SkewSymmExt, VecToQuatExt},
    interp::{
        gauss_legendre_lobotto_points, lagrange_polynomial_derivative,
        quaternion_from_tangent_twist,
    },
};
use nalgebra::{
    DMatrix, DVector, Dyn, Matrix3, Matrix3xX, Matrix4xX, Matrix6xX, OMatrix, Quaternion,
    UnitQuaternion, Vector3, Vector4, Vector6, U7,
};

type Matrix7xX = OMatrix<f64, U7, Dyn>;

#[derive(Debug, Clone)]
pub struct State {
    q: Matrix7xX,            // displacement
    q_delta: Matrix6xX<f64>, // change in displacement
    v: Matrix6xX<f64>,       // velocity
    vd: Matrix6xX<f64>,      // acceleration
    a: Matrix6xX<f64>,       // algorithm acceleration
}

impl State {
    pub fn new(num_nodes: usize) -> Self {
        State {
            q: Matrix7xX::zeros(num_nodes),
            q_delta: Matrix6xX::zeros(num_nodes),
            v: Matrix6xX::zeros(num_nodes),
            vd: Matrix6xX::zeros(num_nodes),
            a: Matrix6xX::zeros(num_nodes),
        }
    }
    pub fn displacement(&self, h: f64) -> (Matrix3xX<f64>, Matrix4xX<f64>) {
        // Create matrices to hold next displacements and rotations
        let mut un: Matrix3xX<f64> = Matrix3xX::zeros(self.q.ncols());
        let mut rn: Matrix4xX<f64> = Matrix4xX::zeros(self.q.ncols());

        // Update displacements
        let u = self.q.fixed_rows::<3>(0);
        let du = self.q_delta.fixed_rows::<3>(0);
        un.copy_from(&(&du * h + &u));

        // Update rotations
        let r = self.q.fixed_rows::<4>(3);
        let dr = self.q_delta.fixed_rows::<3>(3);
        for i in 0..r.ncols() {
            // Get current rotation as a quaternion
            let R = Vector4::from(r.column(i)).as_unit_quaternion();

            // Get change in rotation as quaternion
            let dR = UnitQuaternion::from_scaled_axis(h * dr.column(i));

            // Compose rotation with delta rotation to get new rotation
            let Rn = R * dR;

            // Update rotation vector
            rn.column_mut(i)
                .copy_from(&Vector4::new(Rn.w, Rn.i, Rn.j, Rn.k));
        }
        (un, rn)
    }
}

pub struct GeneralizedAlphaSolver {
    t: f64, // current time (sec)
    h: f64, // time step (sec)
    alpha_m: f64,
    alpha_f: f64,
    gamma: f64,
    beta: f64,
    gamma_prime: f64,
    beta_prime: f64,
    num_state_nodes: usize,
    num_constraint_nodes: usize,
    state: State,
    St: DMatrix<f64>, // iteration matrix
    R: DVector<f64>,  // residual vector
    ndofs: usize,
}

impl GeneralizedAlphaSolver {
    pub fn new(
        num_state_nodes: usize,
        num_constraint_nodes: usize,
        rho_inf: f64,
        t0: f64,
        h: f64,
    ) -> Self {
        // Generalized alpha parameters
        let alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f = rho_inf / (rho_inf + 1.);
        let gamma = 0.5 + alpha_f - alpha_m;
        let beta = 0.25 * (gamma + 0.5).powi(2);
        let beta_prime = (1. - alpha_m) / (h * h * beta * (1. - alpha_f));
        let gamma_prime = gamma / (h * beta);

        // Number of DOFs in full system with constraints
        let ndofs = (num_state_nodes + num_constraint_nodes) * 6;

        // Generalized Alpha structure
        GeneralizedAlphaSolver {
            t: t0,
            h,
            alpha_m,
            alpha_f,
            gamma,
            beta,
            gamma_prime,
            beta_prime,
            num_state_nodes,
            num_constraint_nodes,
            state: State::new(num_state_nodes),
            St: DMatrix::zeros(ndofs, ndofs),
            R: DVector::zeros(ndofs),
            ndofs,
        }
    }

    pub fn predict_next_state(&mut self) -> State {
        let mut state_next = self.state.clone();

        // Initialize acceleration to zero
        state_next.vd.fill(0.);

        // Calculate algorithmic acceleration
        state_next.a =
            (self.alpha_f * &self.state.vd - self.alpha_m * &self.state.a) / (1. - self.alpha_m);

        // Calculate velocity
        state_next.v = &self.state.v
            + self.h * (1. - self.gamma) * &self.state.a
            + self.gamma * self.h * &state_next.a;

        // Calculate change in displacement/rotation (average velocity)
        state_next.q_delta = &self.state.v
            + (0.5 - self.beta) * self.h * &self.state.a
            + self.beta * self.h * &state_next.a;

        state_next
    }

    fn tangent_operator(&self) -> DMatrix<f64> {
        let mut T: DMatrix<f64> =
            DMatrix::zeros(self.num_state_nodes * 6, self.num_state_nodes * 6);
        for i in 0..self.num_state_nodes {
            // Translation
            T.fixed_view_mut::<3, 3>(i * 6, i * 6)
                .copy_from(&Matrix3::identity());

            // Rotation
            // let rot_vec: Vector3<f64> =
            //     Vector3::from(self.h * self.q_delta.fixed_view::<3, 1>(3, i));
            // let phi = rot_vec.magnitude();
            // let psi_t: Matrix3<f64> = rot_vec.skew_symmetric_matrix();
            // let T_SO3: Matrix3<f64> = if phi != 0. {
            //     Matrix3::identity()
            //         + (phi.cos() - 1.) / phi.powi(2) * psi_t
            //         + (1. - phi.sin() / phi) / phi.powi(2) * (psi_t * psi_t)
            // } else {
            //     Matrix3::identity()
            // };
            // T.fixed_view_mut::<3, 3>(i * 6 + 3, i * 6 + 3)
            //     .copy_from(&T_SO3);
            T.fixed_view_mut::<3, 3>(i * 6 + 3, i * 6 + 3)
                .copy_from(&Matrix3::identity());
        }
        T
    }
    pub fn solve_time_step(&mut self, elem: &mut Element) -> Option<usize> {
        // Number of degrees of freedom
        let num_node_dofs = self.num_state_nodes * 6;
        let num_constraint_dofs = self.num_constraint_nodes * 6;

        // Predict the next step
        let mut state_next = self.predict_next_state();

        // Initialize lambda
        let mut lambda: DVector<f64> = DVector::zeros(num_constraint_dofs);

        // Convergence iterations
        for i in 0..20 {
            // Use q_delta to predict next displacement and rotation
            let (u, r) = state_next.displacement(self.h);

            // Apply updated displacements to element
            elem.set_displacement(&u, &r);

            // Get constraints_gradient_matrix
            let B: Matrix6xX<f64> = elem.constraints_gradient_matrix();

            // Get residual vector
            let R: DVector<f64> = elem.residual_vector();

            // Get residual constraints vector
            let R_C: DVector<f64> = B.transpose() * &lambda;

            // Get constraint_residual_vector
            let Phi: Vector6<f64> = elem.constraint_residual_vector();

            // Assemble the residual vector
            self.R.fill(0.);
            self.R.rows_mut(0, num_node_dofs).add_assign(&(&R + &R_C));
            self.R
                .rows_mut(num_node_dofs, num_constraint_dofs)
                .add_assign(&Phi);

            // Check for convergence
            let residual_error = self.R.norm();
            if residual_error < 1e-6 {
                state_next.a += (1. - self.alpha_f) / (1. - self.alpha_m) * &state_next.vd;
                self.state = state_next;
                return Some(i);
            }

            // Static iteration matrix
            let K_FE: DMatrix<f64> = elem.stiffness_matrix();

            // Constraint contribution to static iteration matrix
            let K_C: DMatrix<f64> = DMatrix::zeros(K_FE.nrows(), K_FE.ncols());

            // Tangent operator
            let T: DMatrix<f64> = self.tangent_operator();

            // Assemble iteration matrix
            self.St.fill(0.);
            // Quadrant 1,1
            self.St
                .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                .add_assign((&K_FE + &K_C) * &T);
            // Quadrant 1,2
            self.St
                .view_mut((num_node_dofs, 0), (num_constraint_dofs, num_node_dofs))
                .add_assign(&B * &T);
            // Quadrant 2,1
            self.St
                .view_mut((0, num_node_dofs), (num_node_dofs, num_constraint_dofs))
                .add_assign(&B.transpose());

            // Solve system
            let x: DVector<f64> = self
                .St
                .clone()
                .lu()
                .solve(&self.R)
                .expect("Matrix is not invertable");

            // Extract delta x and delta lambda from system solution
            let x_delta: Matrix6xX<f64> =
                Matrix6xX::from_column_slice(x.rows(0, num_node_dofs).as_slice());
            let lambda_delta: DVector<f64> =
                DVector::from(x.rows(num_node_dofs, num_constraint_dofs));

            // update delta q
            state_next.q_delta -= &x_delta / self.h;
            state_next.v -= self.gamma_prime * &x_delta;
            state_next.a -= self.beta_prime * &x_delta;
            lambda -= lambda_delta;
        }
        None
    }
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;
    use nalgebra::{
        DMatrix, DVector, Matrix3, Matrix3xX, Matrix4xX, Matrix6, Matrix6xX, UnitQuaternion,
        Vector3, Vector4,
    };

    use crate::{
        element::{
            gebt::{Material, Nodes, QuatExt, Section},
            interp::{
                gauss_legendre_lobotto_points, lagrange_polynomial_derivative,
                quaternion_from_tangent_twist,
            },
            quadrature::Quadrature,
        },
        solver::GeneralizedAlphaSolver,
    };

    #[test]
    fn test_element_initial_disp() {
        //----------------------------------------------------------------------
        // Initial configuration
        //----------------------------------------------------------------------

        let xi: DVector<f64> = DVector::from_vec(gauss_legendre_lobotto_points(4));
        let s: DVector<f64> = xi.add_scalar(1.) / 2.;
        let num_nodes = s.len();

        let fx = |s: f64| -> f64 { 10. * s };
        let fz = |s: f64| -> f64 { 0. };
        let fy = |s: f64| -> f64 { 0. };
        let ft = |s: f64| -> f64 { 0. };

        // Node initial position
        let x0: Matrix3xX<f64> = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![fx(si), fy(si), fz(si)]),
        );

        // Node initial rotation
        let interp_deriv: DMatrix<f64> = DMatrix::from_iterator(
            num_nodes,
            num_nodes,
            s.iter()
                .flat_map(|&si| lagrange_polynomial_derivative(si, s.as_slice())),
        );
        let mut tangent: Matrix3xX<f64> = &x0 * interp_deriv;
        for mut c in tangent.column_iter_mut() {
            c.normalize_mut();
        }

        let r0: Matrix4xX<f64> = Matrix4xX::from_columns(
            s.iter()
                .zip(tangent.column_iter())
                .map(|(&si, tan)| {
                    quaternion_from_tangent_twist(&Vector3::from(tan), ft(si)).to_vector()
                })
                .collect::<Vec<Vector4<f64>>>()
                .as_slice(),
        );

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

        // Get xyz displacements at node locations
        let u: Matrix3xX<f64> = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![ux(si), uy(si), uz(si)]),
        );

        // Get matrix describing the nodal rotation
        let r: Matrix4xX<f64> = Matrix4xX::from_columns(
            &s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).to_vector())
                .collect_vec(),
        );

        //----------------------------------------------------------------------
        // Material
        //----------------------------------------------------------------------

        // Create material
        let mat = Material {
            eta: Vector3::zeros(),
            mass: Matrix6::identity(),
            stiffness: Matrix6::from_row_slice(&vec![
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

        //----------------------------------------------------------------------
        // Create element
        //----------------------------------------------------------------------

        // Create nodes structure
        let nodes = Nodes::new(&s, &xi, &x0, &r0);

        // Quadrature rule
        let gq = Quadrature::gauss(7);

        // Create element from nodes
        let mut elem = nodes.element(&gq, &sections);

        //----------------------------------------------------------------------
        // Test solve of element with no initial displacement
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver = GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0);

        // Solve time step
        // let num_conv_iter = solver
        //     .solve_time_step(&mut elem)
        //     .expect("solution failed to converge");
        // assert_eq!(num_conv_iter, 0);

        //----------------------------------------------------------------------
        // Test solve of element with initial displacement
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver = GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0);

        // Populate initial state
        solver.state.q.fixed_rows_mut::<3>(0).copy_from(&u);
        solver.state.q.fixed_rows_mut::<4>(3).copy_from(&r);

        // Solve time step
        // let num_conv_iter = solver
        //     .solve_time_step(&mut elem)
        //     .expect("solution failed to converge");
        // assert_eq!(num_conv_iter, 2);

        //----------------------------------------------------------------------
        // Test solve of element with applied load
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver = GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0);

        // Create force matrix, apply 150 lbs force in z direction of last node
        let mut forces: Matrix6xX<f64> = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = 150.;
        elem.apply_force(&forces);

        let num_conv_iter = solver
            .solve_time_step(&mut elem)
            .expect("solution failed to converge");
    }
}
