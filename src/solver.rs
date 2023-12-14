#![allow(non_snake_case)]

use crate::prelude::*;

use crate::element::gebt::Element;

#[derive(Debug, Clone)]
pub struct StepConfig {
    pub h: f64, // time step size (sec)
    pub alpha_m: f64,
    pub alpha_f: f64,
    pub gamma: f64,
    pub beta: f64,
    pub gamma_prime: f64,
    pub beta_prime: f64,
}

impl StepConfig {
    /// return a structure containing the generalized alpha time stepping parameters
    pub fn new(rho_inf: f64, h: f64) -> Self {
        let alpha_m: f64 = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f: f64 = rho_inf / (rho_inf + 1.);
        let gamma: f64 = 0.5 + alpha_f - alpha_m;
        let beta: f64 = 0.25 * (gamma + 0.5).powi(2);
        let beta_prime: f64 = (1. - alpha_m) / (h * h * beta * (1. - alpha_f));
        let gamma_prime: f64 = gamma / (h * beta);
        StepConfig {
            h,
            alpha_m,
            alpha_f,
            gamma,
            beta,
            beta_prime,
            gamma_prime,
        }
    }
    /// value for conditioning the solver system
    pub fn conditioning_value(&self) -> f64 {
        self.beta * self.h * self.h
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub t: f64,             // time
    q: Matrix7xX,           // displacement
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
    /// Returns structure predicting the state at the end of the next step
    pub fn predict_next(&self, cfg: &StepConfig) -> State {
        // Calculate algorithmic acceleration
        let a: Matrix6xX = (cfg.alpha_f * &self.vd - cfg.alpha_m * &self.a) / (1. - cfg.alpha_m);
        State {
            // Get next state time
            t: self.t + cfg.h,
            // Initial displacements are final displacements from previous step
            q: self.displacement(cfg),
            // Calculate change in displacement/rotation (average velocity)
            q_delta: &self.v + (0.5 - cfg.beta) * cfg.h * &self.a + cfg.beta * cfg.h * &a,
            // Calculate velocity
            v: &self.v + cfg.h * (1. - cfg.gamma) * &self.a + cfg.gamma * cfg.h * &a,
            // Initialize acceleration to zero
            vd: Matrix6xX::zeros(self.vd.ncols()),
            a,
        }
    }
    /// Updates predicted state during a convergence iteration
    pub fn update(&mut self, delta_x: &Matrix6xX, cfg: &StepConfig) {
        self.q_delta += delta_x / cfg.h;
        self.v += cfg.gamma_prime * delta_x;
        self.vd += cfg.beta_prime * delta_x;
    }
    /// Updates the algorithmic acceleration at the end of a time step
    pub fn update_algorithmic_acceleration(&mut self, cfg: &StepConfig) {
        self.a += (1. - cfg.alpha_f) / (1. - cfg.alpha_m) * &self.vd;
    }
    pub fn set_displacement(&mut self, Q: &Matrix7xX) {
        self.q.copy_from(Q);
    }
    /// Returns state displacement matrix (initial displacement + delta displacement)
    pub fn displacement(&self, cfg: &StepConfig) -> Matrix7xX {
        let mut Q: Matrix7xX = Matrix7xX::zeros(self.q.ncols());

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * cfg.h;

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
    /// Returns state velocity matrix
    pub fn velocity(&self) -> Matrix6xX {
        self.v.clone()
    }
    /// Returns state acceleration matrix
    pub fn acceleration(&self) -> Matrix6xX {
        self.vd.clone()
    }
    /// Returns tangent operator matrix for modifying iteration matrix
    pub fn tangent_operator(&self, cfg: &StepConfig) -> MatrixN {
        let mut T: MatrixN = MatrixN::zeros(self.q_delta.len(), self.q_delta.len());

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * cfg.h;

        // Loop through nodes
        for i in 0..self.q_delta.ncols() {
            // Translation
            let T_R3: Matrix3 = Matrix3::identity();
            T.fixed_view_mut::<3, 3>(i * 6, i * 6).copy_from(&T_R3);

            // Rotation
            let T_SO3: Matrix3 = Vector3::from(qd.fixed_view::<3, 1>(3, i)).tangent_matrix();
            T.fixed_view_mut::<3, 3>(i * 6 + 3, i * 6 + 3)
                .copy_from(&T_SO3);
        }
        T
    }
}

pub struct GeneralizedAlphaSolver {
    pub state: State,
    pub step_config: StepConfig,
    num_state_nodes: usize,
    num_constraint_nodes: usize,
    St: MatrixD, // iteration matrix
    R: VectorD,  // residual vector
    gravity: Vector3,
    DL_diag: VectorD, // left conditioning vector
    DR_diag: VectorD, // right conditioning vector
    DL: MatrixD,      // left conditioning matrix
    DR: MatrixD,      // right conditioning matrix
}

impl GeneralizedAlphaSolver {
    pub fn new(
        num_state_nodes: usize,
        num_constraint_nodes: usize,
        step_config: &StepConfig,
        t0: f64,
        gravity: Vector3,
    ) -> Self {
        let num_state_dofs = num_state_nodes * 6;
        let num_constraint_dofs = num_constraint_nodes * 6;
        // Number of DOFs in full system with constraints
        let ndofs = num_state_dofs + num_constraint_dofs;

        // Solution conditioning
        let cond_scale = step_config.conditioning_value();
        let mut DL_diag: VectorD = VectorD::zeros(ndofs).add_scalar(1.);
        DL_diag.rows_mut(0, num_state_dofs).fill(cond_scale);
        let DL: MatrixD = MatrixD::from_diagonal(&DL_diag);
        let mut DR_diag: VectorD = VectorD::zeros(ndofs).add_scalar(1.);
        DR_diag
            .rows_mut(num_state_dofs, num_constraint_dofs)
            .fill(1. / (cond_scale));
        let DR: MatrixD = MatrixD::from_diagonal(&DR_diag);

        // Generalized Alpha structure
        GeneralizedAlphaSolver {
            step_config: step_config.clone(),
            num_state_nodes,
            num_constraint_nodes,
            state: State::new(num_state_nodes, t0),
            St: MatrixN::zeros(ndofs, ndofs),
            R: VectorN::zeros(ndofs),
            gravity,
            DL_diag,
            DR_diag,
            DL,
            DR,
        }
    }

    pub fn step(&mut self, elem: &mut Element) -> Option<Vec<f64>> {
        let mut res_norm: Vec<f64> = Vec::new();

        // Number of degrees of freedom
        let num_node_dofs = self.num_state_nodes * 6;
        let num_constraint_dofs = self.num_constraint_nodes * 6;

        // Predict the next step
        let mut state_next = self.state.predict_next(&self.step_config);

        // Initialize lambda
        let mut lambda: VectorD = VectorD::zeros(num_constraint_dofs);

        let mut energy_increment_ref: f64 = 0.;

        // Convergence iterations
        for i in 0..1000 {
            // Predict next displacement and velocity
            let Q: Matrix7xX = state_next.displacement(&self.step_config);
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
            self.R.rows_mut(0, num_node_dofs).add_assign(&(R_FE + F_C));
            self.R
                .rows_mut(num_node_dofs, num_constraint_dofs)
                .add_assign(&Phi);

            // Element matrices
            let M: MatrixD = elem.M();
            let G: MatrixD = elem.G();
            let K_FE: MatrixD = elem.K_FE();

            // Constraint contribution to static iteration matrix
            let K_C: MatrixD = MatrixD::zeros(K_FE.nrows(), K_FE.ncols());

            // Tangent operator
            let T: MatrixD = state_next.tangent_operator(&self.step_config);

            // Assemble iteration matrix
            self.St.fill(0.);
            // Quadrant 1,1
            self.St
                .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                .add_assign(
                    M * self.step_config.beta_prime
                        + G * self.step_config.gamma_prime
                        + (K_FE + K_C) * &T,
                );
            // Quadrant 1,2
            self.St
                .view_mut((num_node_dofs, 0), (num_constraint_dofs, num_node_dofs))
                .add_assign(&B * &T);
            // Quadrant 2,1
            self.St
                .view_mut((0, num_node_dofs), (num_node_dofs, num_constraint_dofs))
                .add_assign(&B.transpose());

            // Condition the iteration matrix
            self.St = &self.DL * &self.St * &self.DR;
            self.R.component_mul_assign(&self.DL_diag);

            // Solve system
            let mut x: VectorD = self
                .St
                .clone()
                .lu()
                .solve(&self.R)
                .expect("Matrix is not invertable");

            // Un-condition the solution vector
            x.component_mul_assign(&self.DR_diag);

            // Extract delta x and delta lambda from system solution
            let delta_x: Matrix6xX =
                -Matrix6xX::from_column_slice(x.rows(0, num_node_dofs).as_slice());
            let lambda_delta: VectorD = -VectorD::from(x.rows(num_node_dofs, num_constraint_dofs));

            // update state and lambda
            state_next.update(&delta_x, &self.step_config);
            lambda += lambda_delta;

            // Check for convergence
            let energy_increment = self.R.dot(&x);
            if i == 0 {
                energy_increment_ref = energy_increment;
            }
            res_norm.push(if energy_increment_ref == 0. {
                0.
            } else {
                energy_increment / energy_increment_ref
            });
            if energy_increment_ref == 0. || (energy_increment / energy_increment_ref < 1e-3) {
                state_next.update_algorithmic_acceleration(&self.step_config);
                self.state = state_next;
                return Some(res_norm);
            }
        }
        None
    }
}
