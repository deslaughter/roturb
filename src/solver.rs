#![allow(non_snake_case)]

use crate::prelude::*;

use crate::element::gebt::Element;

#[derive(Debug, Clone)]
pub struct State {
    pub t: f64,             // time
    pub q: Matrix7xX,       // displacement
    pub q_delta: Matrix6xX, // change in displacement
    pub v: Matrix6xX,       // velocity
    pub vd: Matrix6xX,      // acceleration
    pub a: Matrix6xX,       // algorithm acceleration
}

impl State {
    pub fn new(num_nodes: usize, t0: f64) -> Self {
        State {
            t: t0,
            q: Matrix7xX::zeros(num_nodes),
            q_delta: Matrix6xX::zeros(num_nodes),
            v: Matrix6xX::zeros(num_nodes),
            vd: Matrix6xX::zeros(num_nodes),
            a: Matrix6xX::zeros(num_nodes),
        }
    }
    pub fn displacement(&self, h: f64) -> Matrix7xX {
        let mut Q: Matrix7xX = Matrix7xX::zeros(self.q.ncols());

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * h;

        // Calculate new displacements
        let mut un = Q.fixed_rows_mut::<3>(0);
        let u = self.q.fixed_rows::<3>(0);
        let du = qd.fixed_rows::<3>(0);
        un.copy_from(&(&u + &du));

        // Calculate new rotations
        let mut rn = Q.fixed_rows_mut::<4>(3);
        let r = self.q.fixed_rows::<4>(3);
        let dr = qd.fixed_rows::<3>(3);
        for i in 0..r.ncols() {
            let R1: UnitQuaternion = Vector4::from(r.column(i)).as_unit_quaternion(); // current rotation
            let R2: UnitQuaternion = UnitQuaternion::from_scaled_axis(dr.column(i)); // change in rotation
            let Rn: UnitQuaternion = R1 * R2; // new rotation
            rn.column_mut(i)
                .copy_from(&Vector4::new(Rn.w, Rn.i, Rn.j, Rn.k));
        }
        Q
    }
    pub fn velocity(&self) -> Matrix6xX {
        self.v.clone()
    }
    pub fn acceleration(&self) -> Matrix6xX {
        self.vd.clone()
    }
    pub fn tangent_operator(&self, h: f64) -> MatrixN {
        let mut T: MatrixN = MatrixN::zeros(self.q_delta.len(), self.q_delta.len());

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * h;

        // Loop through nodes
        for i in 0..self.q_delta.ncols() {
            // Translation
            let T_R3: Matrix3 = Matrix3::identity();
            T.fixed_view_mut::<3, 3>(i * 6, i * 6).copy_from(&T_R3);

            // Rotation
            let psi: Vector3 = Vector3::from(qd.fixed_view::<3, 1>(3, i));
            let phi = psi.magnitude();
            let T_SO3: Matrix3 = if phi == 0. {
                Matrix3::identity()
            } else {
                Matrix3::identity()
                    + (1. - phi.cos()) / phi.powi(2) * psi.tilde()
                    + (1. - phi.sin() / phi) / phi.powi(2) * (psi.tilde() * psi.tilde())
            };
            T.fixed_view_mut::<3, 3>(i * 6 + 3, i * 6 + 3)
                .copy_from(&T_SO3);
        }
        T
    }
}

pub struct GeneralizedAlphaSolver {
    pub state: State,
    h: f64, // time step (sec)
    alpha_m: f64,
    alpha_f: f64,
    gamma: f64,
    beta: f64,
    gamma_prime: f64,
    beta_prime: f64,
    num_state_nodes: usize,
    num_constraint_nodes: usize,
    St: MatrixD, // iteration matrix
    R: VectorD,  // residual vector
    gravity: Vector3,
}

impl GeneralizedAlphaSolver {
    pub fn new(
        num_state_nodes: usize,
        num_constraint_nodes: usize,
        rho_inf: f64,
        t0: f64,
        h: f64,
        gravity: Vector3,
    ) -> Self {
        // Generalized alpha parameters
        let alpha_m: f64 = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f: f64 = rho_inf / (rho_inf + 1.);
        let gamma: f64 = 0.5 + alpha_f - alpha_m;
        let beta: f64 = 0.25 * (gamma + 0.5).powi(2);
        let beta_prime: f64 = (1. - alpha_m) / (h * h * beta * (1. - alpha_f));
        let gamma_prime: f64 = gamma / (h * beta);

        // Number of DOFs in full system with constraints
        let ndofs = (num_state_nodes + num_constraint_nodes) * 6;

        // Generalized Alpha structure
        GeneralizedAlphaSolver {
            h,
            alpha_m,
            alpha_f,
            gamma,
            beta,
            gamma_prime,
            beta_prime,
            num_state_nodes,
            num_constraint_nodes,
            state: State::new(num_state_nodes, t0),
            St: MatrixN::zeros(ndofs, ndofs),
            R: VectorN::zeros(ndofs),
            gravity,
        }
    }

    pub fn predict_next_state(&mut self, h: f64) -> State {
        let mut state_next = self.state.clone();

        // Increment time
        state_next.t += h;

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

    pub fn solve_time_step(&mut self, elem: &mut Element) -> Option<usize> {
        // Number of degrees of freedom
        let num_node_dofs = self.num_state_nodes * 6;
        let num_constraint_dofs = self.num_constraint_nodes * 6;

        // Predict the next step
        let mut state_next = self.predict_next_state(self.h);

        // Initialize lambda
        let mut lambda: VectorD = VectorD::zeros(num_constraint_dofs);

        // Convergence iterations
        for i in 0..20 {
            // Predict next displacement and velocity
            let Q: Matrix7xX = state_next.displacement(self.h);
            let V: Matrix6xX = state_next.velocity();
            let A: Matrix6xX = state_next.acceleration();

            // Apply updated displacements to element
            elem.update_states(&Q, &V, &A, &self.gravity);

            // Get constraints_gradient_matrix
            let B: Matrix6xX = elem.constraints_gradient_matrix();

            // Get element residual vector
            let R_FE: VectorD = elem.R_FE();

            // Get constraints residual vector
            let F_C: VectorD = B.transpose() * &lambda;

            // Get constraint_residual_vector
            let Phi: Vector6 = elem.constraint_residual_vector();

            // Assemble the residual vector
            self.R.fill(0.);
            self.R
                .rows_mut(0, num_node_dofs)
                .add_assign(&(&R_FE + &F_C));
            self.R
                .rows_mut(num_node_dofs, num_constraint_dofs)
                .add_assign(&Phi);

            // Check for convergence
            let residual_error = self.R.norm();
            if residual_error < 1e-6 {
                state_next.a += (1. - self.alpha_f) / (1. - self.alpha_m) * &state_next.vd;
                state_next.q = Q;
                self.state = state_next;
                return Some(i);
            }

            // Element matrices
            let M: MatrixD = elem.M();
            let G: MatrixD = elem.G();
            let K_FE: MatrixD = elem.K_FE();

            // Constraint contribution to static iteration matrix
            let K_C: MatrixD = MatrixD::zeros(K_FE.nrows(), K_FE.ncols());

            // Tangent operator
            let T: MatrixD = state_next.tangent_operator(self.h);

            // Assemble iteration matrix
            self.St.fill(0.);
            // Quadrant 1,1
            self.St
                .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                .add_assign(M * self.beta_prime + G * self.gamma_prime + (K_FE + K_C) * &T);
            // Quadrant 1,2
            self.St
                .view_mut((num_node_dofs, 0), (num_constraint_dofs, num_node_dofs))
                .add_assign(&B * &T);
            // Quadrant 2,1
            self.St
                .view_mut((0, num_node_dofs), (num_node_dofs, num_constraint_dofs))
                .add_assign(&B.transpose());

            // Solve system
            let x: VectorD = self
                .St
                .clone()
                .lu()
                .solve(&self.R)
                .expect("Matrix is not invertable");

            // Extract delta x and delta lambda from system solution
            let x_delta: Matrix6xX =
                Matrix6xX::from_column_slice(x.rows(0, num_node_dofs).as_slice());
            let lambda_delta: VectorD = VectorD::from(x.rows(num_node_dofs, num_constraint_dofs));

            // update delta q
            state_next.q_delta -= &x_delta / self.h;
            state_next.v -= self.gamma_prime * &x_delta;
            state_next.vd -= self.beta_prime * &x_delta;
            lambda -= lambda_delta;
        }
        None
    }
}
