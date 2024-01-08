#![allow(non_snake_case)]

use crate::prelude::*;

use crate::element::gebt::Element;

#[derive(Debug, Clone)]
pub struct State {
    pub num_system_nodes: usize,     // number of system nodes
    pub num_constraint_nodes: usize, // number of constraint nodes
    pub q_delta: Matrix6xX,          // change in displacement
    pub q_prev: Matrix7xX,           // displacement at end of previous step
    pub q: Matrix7xX,                // displacement
    pub v: Matrix6xX,                // velocity
    pub vd: Matrix6xX,               // acceleration
    pub a: Matrix6xX,                // algorithm acceleration
    pub lambda: Matrix6xX,           // constraints
}

impl State {
    pub fn new(num_system_nodes: usize, num_constraint_nodes: usize) -> Self {
        State {
            num_system_nodes,
            num_constraint_nodes,
            q_delta: Matrix6xX::zeros(num_system_nodes),
            q_prev: Matrix7xX::zeros(num_system_nodes),
            q: Matrix7xX::zeros(num_system_nodes),
            v: Matrix6xX::zeros(num_system_nodes),
            vd: Matrix6xX::zeros(num_system_nodes),
            a: Matrix6xX::zeros(num_system_nodes),
            lambda: Matrix6xX::zeros(num_constraint_nodes),
        }
    }

    pub fn new_with_initial_state(
        num_system_nodes: usize,
        num_constraint_nodes: usize,
        Q: &Matrix7xX,
        V: &Matrix6xX,
        A: &Matrix6xX,
    ) -> Self {
        State {
            num_system_nodes,
            num_constraint_nodes,
            q_delta: Matrix6xX::zeros(num_system_nodes),
            q_prev: Q.clone(),
            q: Matrix7xX::zeros(num_system_nodes),
            v: V.clone(),
            vd: A.clone(),
            a: Matrix6xX::zeros(num_system_nodes),
            lambda: Matrix6xX::zeros(num_constraint_nodes),
        }
    }

    /// Returns structure predicting the state at the end of the next step
    pub fn predict_next(&self, h: f64, beta: f64, gamma: f64, alpha_m: f64, alpha_f: f64) -> State {
        let mut ns = self.clone();
        ns.a = (alpha_f * &self.vd - alpha_m * &self.a) / (1. - alpha_m);
        ns.q_delta = &self.v + (0.5 - beta) * h * &self.a + beta * h * &ns.a;
        ns.q_prev.copy_from(&self.q);
        ns.update_q(h);
        ns.v = &self.v + h * (1. - gamma) * &self.a + gamma * h * &ns.a;
        ns.vd.fill(0.);
        ns.lambda.fill(0.);
        ns
    }

    /// Updates state during a dynamic solve
    pub fn update_dynamic(&mut self, delta: &Matrix6xX, h: f64, beta_prime: f64, gamma_prime: f64) {
        let x_delta = delta.columns(0, self.num_system_nodes);
        self.q_delta += x_delta / h;
        self.v += gamma_prime * x_delta;
        self.vd += beta_prime * x_delta;
        self.update_q(h);
        let lambda_delta = delta.columns(self.num_system_nodes, self.num_constraint_nodes);
        self.lambda += lambda_delta;
    }

    // Updates state during a static solve
    pub fn update_static(&mut self, delta: &Matrix6xX, h: f64) {
        let x_delta = delta.columns(0, self.num_system_nodes);
        self.q_delta += x_delta / h;
        self.update_q(h);
        let lambda_delta = delta.columns(self.num_system_nodes, self.num_constraint_nodes);
        self.lambda += lambda_delta;
    }

    /// Updates the algorithmic acceleration at the end of a time step
    pub fn update_algorithmic_acceleration(&mut self, alpha_m: f64, alpha_f: f64) {
        self.a += (1. - alpha_f) / (1. - alpha_m) * &self.vd;
    }

    // Update q based on changes to q_delta
    fn update_q(&mut self, h: f64) {
        // Multiply q_delta by h to get change in state
        let h_q_delta: Matrix6xX = h * &self.q_delta;

        // Calculate new displacement
        let u = self.q_prev.fixed_rows::<3>(0);
        let du = h_q_delta.fixed_rows::<3>(0);
        self.q.fixed_rows_mut::<3>(0).copy_from(&(&u + &du));

        // Calculate new rotation
        let mut rn = self.q.fixed_rows_mut::<4>(3);
        let r = self.q_prev.fixed_rows::<4>(3);
        let dr = h_q_delta.fixed_rows::<3>(3);
        for (i, mut c) in rn.column_iter_mut().enumerate() {
            let R1: UnitQuaternion = Vector4::from(r.column(i)).as_unit_quaternion(); // current rotation
            let R2: UnitQuaternion = UnitQuaternion::from_scaled_axis(dr.column(i)); // change in rotation
            c.copy_from(&(R2 * R1).wijk());
        }
    }

    /// Returns tangent operator matrix for modifying iteration matrix
    pub fn tangent_operator(&self, h: f64) -> MatrixN {
        let mut T: MatrixN = MatrixN::zeros(self.q_delta.len(), self.q_delta.len());

        // Get change in delta multiplied by time step
        let h_q_delta: Matrix6xX = &self.q_delta * h;

        // Loop through nodes
        for (i, c) in h_q_delta.column_iter().enumerate() {
            // Translation
            let T_R3: Matrix3 = Matrix3::identity();
            T.fixed_view_mut::<3, 3>(i * 6, i * 6).copy_from(&T_R3);

            // Rotation
            let T_SO3: Matrix3 = c.fixed_rows::<3>(3).clone_owned().tangent_matrix();
            T.fixed_view_mut::<3, 3>(i * 6 + 3, i * 6 + 3)
                .copy_from(&T_SO3);
        }
        T
    }
}

pub struct GeneralizedAlphaSolver {
    num_state_nodes: usize,
    num_constraint_nodes: usize,
    pub h: f64, // time step size (sec)
    pub alpha_m: f64,
    pub alpha_f: f64,
    pub gamma: f64,
    pub beta: f64,
    pub gamma_prime: f64,
    pub beta_prime: f64,
    St: MatrixD, // iteration matrix
    R: VectorD,  // residual vector
    gravity: Vector3,
    DL_diag: VectorD, // left conditioning vector
    DR_diag: VectorD, // right conditioning vector
    DL: MatrixD,      // left conditioning matrix
    DR: MatrixD,      // right conditioning matrix
    is_dynamic_solve: bool,
}

impl GeneralizedAlphaSolver {
    pub fn new(
        num_system_nodes: usize,
        num_constraint_nodes: usize,
        rho_inf: f64,
        h: f64,
        gravity: Vector3,
        is_dynamic_solve: bool,
    ) -> Self {
        let num_state_dofs = num_system_nodes * 6;
        let num_constraint_dofs = num_constraint_nodes * 6;
        // Number of DOFs in full system with constraints
        let ndofs = num_state_dofs + num_constraint_dofs;

        // Generalized alpha parameters
        let alpha_m: f64 = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f: f64 = rho_inf / (rho_inf + 1.);
        let gamma: f64 = 0.5 + alpha_f - alpha_m;
        let beta: f64 = 0.25 * (gamma + 0.5).powi(2);
        let beta_prime: f64 = (1. - alpha_m) / (h * h * beta * (1. - alpha_f));
        let gamma_prime: f64 = gamma / (h * beta);

        // Solution conditioning
        let cond_scale = beta * h.powi(2);
        let mut DL_diag: VectorD = VectorD::zeros(ndofs).add_scalar(1.);
        DL_diag.rows_mut(0, num_state_dofs).fill(cond_scale);
        let mut DR_diag: VectorD = VectorD::zeros(ndofs).add_scalar(1.);
        DR_diag
            .rows_mut(num_state_dofs, num_constraint_dofs)
            .fill(1. / (cond_scale));
        let DL: MatrixD = MatrixD::from_diagonal(&DL_diag);
        let DR: MatrixD = MatrixD::from_diagonal(&DR_diag);

        // Generalized Alpha structure
        GeneralizedAlphaSolver {
            num_state_nodes: num_system_nodes,
            num_constraint_nodes,
            h,
            alpha_m,
            alpha_f,
            gamma,
            beta,
            beta_prime,
            gamma_prime,
            St: MatrixN::zeros(ndofs, ndofs),
            R: VectorN::zeros(ndofs),
            gravity,
            DL_diag,
            DR_diag,
            DL,
            DR,
            is_dynamic_solve,
        }
    }

    pub fn step(&mut self, elem: &mut Element, state: &State) -> Option<(State, Vec<IterData>)> {
        let mut iter_data: Vec<IterData> = Vec::new();

        // Number of degrees of freedom
        let num_node_dofs = self.num_state_nodes * 6;
        let num_constraint_dofs = self.num_constraint_nodes * 6;

        // Predict the next step
        let mut state_next =
            state.predict_next(self.h, self.beta, self.gamma, self.alpha_m, self.alpha_f);

        // Initialize reference energy increment
        let mut energy_increment_ref: f64 = 0.;

        // Convergence iterations
        for i in 0..50 {
            // If no constraints, prescribe the root displacement on node 1
            if self.num_constraint_nodes == 0 {
                state_next.q.column_mut(0).copy_from(&elem.q_root);
            }

            // Update state in element
            elem.update_states(&state_next.q, &state_next.v, &state_next.vd, &self.gravity);

            // Get element residual vector, stiffness, damping, and mass matrices
            let R_FE: VectorD = elem.R_FE();
            let K_FE: MatrixD = elem.K_FE();
            let M: MatrixD = elem.M();
            let G: MatrixD = elem.G();

            // Constraint contribution to static iteration matrix
            let K_C: MatrixD = MatrixD::zeros(K_FE.nrows(), K_FE.ncols());

            // Tangent operator
            let T: MatrixD = state_next.tangent_operator(self.h);

            // Assemble the residual vector
            self.R.fill(0.);
            self.R.rows_mut(0, num_node_dofs).add_assign(&(R_FE));

            // Assemble iteration matrix
            self.St.fill(0.);
            // Quadrant 1,1
            if self.is_dynamic_solve {
                self.St
                    .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                    .add_assign(M * self.beta_prime + G * self.gamma_prime + (K_FE + K_C) * &T);
            } else {
                self.St
                    .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                    .add_assign((K_FE + K_C) * &T);
            }

            // If there are constraint nodes
            if self.num_constraint_nodes > 0 {
                // Get constraints_gradient_matrix
                let B: Matrix6xX = elem.constraints_gradient_matrix();

                // Get constraints residual vector
                let F_C: VectorD =
                    B.transpose() * &VectorD::from_column_slice(state_next.lambda.as_slice());

                // Get constraint_residual_vector
                let Phi: Vector6 = elem.constraint_residual_vector();

                self.R
                    .rows_mut(num_node_dofs, num_constraint_dofs)
                    .add_assign(&Phi);
                self.R.rows_mut(0, num_node_dofs).add_assign(&(F_C));

                // Quadrant 1,2
                self.St
                    .view_mut((num_node_dofs, 0), (num_constraint_dofs, num_node_dofs))
                    .add_assign(&B * &T);
                // Quadrant 2,1

                self.St
                    .view_mut((0, num_node_dofs), (num_node_dofs, num_constraint_dofs))
                    .add_assign(&B.transpose());
            }

            // Condition the iteration matrix and residuals
            let St: MatrixD = &self.DL * &self.St * &self.DR;
            let R: VectorD = self.R.component_mul(&self.DL_diag);

            // Solve system
            let mut x: VectorD = VectorD::zeros(num_node_dofs);
            if num_constraint_dofs == 0 {
                let reduced_dofs = num_node_dofs - 6;
                let xx: VectorD = St
                    .view((6, 6), (reduced_dofs, reduced_dofs))
                    .lu()
                    .solve(&R.rows(6, reduced_dofs))
                    .expect("Matrix is not invertable");
                x.rows_mut(6, reduced_dofs).copy_from(&xx);
            } else {
                x = St.lu().solve(&R).expect("Matrix is not invertable");
            }

            // If the solution contains NaNs, return
            if x.iter().any(|&v| v.is_nan()) {
                print!("NaNs in solve\n");
                return None;
            }

            // Un-condition the solution vector
            let delta_x: Matrix6xX =
                Matrix6xX::from_column_slice((-x.component_mul(&self.DR_diag)).as_slice());

            // update state and lambda
            if self.is_dynamic_solve {
                state_next.update_dynamic(&delta_x, self.h, self.beta_prime, self.gamma_prime);
            } else {
                state_next.update_static(&delta_x, self.h);
            }

            // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_elem.json", i)).unwrap();
            // serde_json::to_writer_pretty(f, elem).expect("fail");
            // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_St.json", i)).unwrap();
            // serde_json::to_writer_pretty(f, &self.St).expect("fail");
            // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_R.json", i)).unwrap();
            // serde_json::to_writer_pretty(f, &self.R).expect("fail");
            // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_x.json", i)).unwrap();
            // serde_json::to_writer_pretty(f, &x).expect("fail");
            // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_q.json", i)).unwrap();
            // serde_json::to_writer_pretty(f, &state_next.q()).expect("fail");

            // Check for convergence
            let energy_increment = self.R.dot(&x).abs();
            iter_data.push(IterData {
                energy_inc: energy_increment,
                residual: self.R.clone(),
                x: x.clone(),
            });
            if i == 0 {
                energy_increment_ref = energy_increment;
            }
            let energy_ratio = energy_increment / energy_increment_ref;
            if energy_increment < 1e-8 || energy_ratio < 1e-5 {
                state_next.update_algorithmic_acceleration(self.alpha_m, self.alpha_f);
                return Some((state_next, iter_data));
            }
        }
        // Solution did not converge
        state_next.update_algorithmic_acceleration(self.alpha_m, self.alpha_f);
        Some((state_next, iter_data))
        // None
    }
}

pub struct IterData {
    pub energy_inc: f64,
    pub residual: VectorD,
    pub x: VectorD,
}
