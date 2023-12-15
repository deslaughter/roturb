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
}

#[derive(Debug, Clone)]
pub struct State {
    cfg: StepConfig,    // time step configuration
    num_nodes: usize,   // number of nodes
    t: f64,             // time
    qp: Matrix7xX,      // displacement at end of previous step
    q_delta: Matrix6xX, // change in displacement
    q: Matrix7xX,       // current displacement
    v: Matrix6xX,       // velocity
    vd: Matrix6xX,      // acceleration
    a: Matrix6xX,       // algorithm acceleration
}

impl State {
    pub fn new(cfg: &StepConfig, num_nodes: usize, t0: f64) -> Self {
        State {
            cfg: cfg.clone(),
            num_nodes,
            t: t0,
            qp: Matrix7xX::zeros(num_nodes),
            q_delta: Matrix6xX::zeros(num_nodes),
            q: Matrix7xX::zeros(num_nodes),
            v: Matrix6xX::zeros(num_nodes),
            vd: Matrix6xX::zeros(num_nodes),
            a: Matrix6xX::zeros(num_nodes),
        }
    }
    pub fn new_with_initial_state(
        cfg: &StepConfig,
        num_nodes: usize,
        t0: f64,
        Q: &Matrix7xX,
        V: &Matrix6xX,
        A: &Matrix6xX,
    ) -> Self {
        State {
            cfg: cfg.clone(),
            num_nodes,
            t: t0,
            qp: Q.clone(),
            q_delta: Matrix6xX::zeros(num_nodes),
            q: Q.clone(),
            v: V.clone(),
            vd: A.clone(),
            a: Matrix6xX::zeros(num_nodes),
        }
    }
    /// Returns structure predicting the state at the end of the next step
    pub fn predict_next(&self) -> State {
        // Calculate algorithmic acceleration
        let a: Matrix6xX =
            (self.cfg.alpha_f * &self.vd - self.cfg.alpha_m * &self.a) / (1. - self.cfg.alpha_m);
        let mut ns = State {
            // Store step configuration
            cfg: self.cfg.clone(),
            num_nodes: self.num_nodes,
            // Get next state time (time at end of step)
            t: self.t + self.cfg.h,
            // Save final displacements from current state as previous displacements
            qp: self.q.clone(),
            // Initialize current displacements to zero
            q: Matrix7xX::zeros(self.q.ncols()),
            // Calculate change in displacement/rotation (average velocity)
            q_delta: &self.v
                + (0.5 - self.cfg.beta) * self.cfg.h * &self.a
                + self.cfg.beta * self.cfg.h * &a,
            // Calculate velocity
            v: &self.v
                + self.cfg.h * (1. - self.cfg.gamma) * &self.a
                + self.cfg.gamma * self.cfg.h * &a,
            // Initialize acceleration to zero
            vd: Matrix6xX::zeros(self.vd.ncols()),
            a,
        };
        // Calculate new displacements
        // (delta_x is zero so q_delta, velocity, and acceleration don't change)
        ns.update(&Matrix6xX::zeros(self.q.ncols()));
        ns
    }
    /// Updates predicted state during a convergence iteration
    pub fn update(&mut self, delta_x: &Matrix6xX) {
        self.q_delta += delta_x / self.cfg.h;
        self.v += self.cfg.gamma_prime * delta_x;
        self.vd += self.cfg.beta_prime * delta_x;

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * self.cfg.h;

        // Calculate new displacement
        let mut un = self.q.fixed_rows_mut::<3>(0);
        let u = self.qp.fixed_rows::<3>(0);
        let du = qd.fixed_rows::<3>(0);
        un.copy_from(&(&u + &du));

        // Calculate new rotation
        let mut rn = self.q.fixed_rows_mut::<4>(3);
        let r = self.qp.fixed_rows::<4>(3);
        let dr = qd.fixed_rows::<3>(3);
        for i in 0..r.ncols() {
            let R1: UnitQuaternion = Vector4::from(r.column(i)).as_unit_quaternion(); // current rotation
            let R2: UnitQuaternion = UnitQuaternion::from_scaled_axis(dr.column(i)); // change in rotation
            let Rn: UnitQuaternion = R1 * R2; // new rotation
            rn.column_mut(i)
                .copy_from(&Vector4::new(Rn.w, Rn.i, Rn.j, Rn.k));
        }
    }
    /// Updates the algorithmic acceleration at the end of a time step
    pub fn update_algorithmic_acceleration(&mut self) {
        self.a += (1. - self.cfg.alpha_f) / (1. - self.cfg.alpha_m) * &self.vd;
    }
    /// Returns state time
    pub fn time(&self) -> f64 {
        self.t
    }
    /// Returns state displacement matrix
    pub fn Q(&self) -> Matrix7xX {
        self.q.clone()
    }
    /// Returns state velocity matrix
    pub fn V(&self) -> Matrix6xX {
        self.v.clone()
    }
    /// Returns state acceleration matrix
    pub fn A(&self) -> Matrix6xX {
        self.vd.clone()
    }

    /// Returns tangent operator matrix for modifying iteration matrix
    pub fn tangent_operator(&self) -> MatrixN {
        let mut T: MatrixN = MatrixN::zeros(self.q_delta.len(), self.q_delta.len());

        // Get change in delta multiplied by time step
        let qd: Matrix6xX = &self.q_delta * self.cfg.h;

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

    /// Returns value for conditioning the solver system
    pub fn conditioning_value(&self) -> f64 {
        self.cfg.beta * self.cfg.h.powi(2)
    }
}

pub struct GeneralizedAlphaSolver {
    pub state: State,
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
        state0: &State,
        gravity: Vector3,
    ) -> Self {
        let num_state_dofs = num_state_nodes * 6;
        let num_constraint_dofs = num_constraint_nodes * 6;
        // Number of DOFs in full system with constraints
        let ndofs = num_state_dofs + num_constraint_dofs;

        // Solution conditioning
        let cond_scale = state0.conditioning_value();
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
            num_state_nodes,
            num_constraint_nodes,
            state: state0.clone(),
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
        let mut state_next = self.state.predict_next();

        // Initialize lambda
        let mut lambda: VectorD = VectorD::zeros(num_constraint_dofs);

        let mut energy_increment_ref: f64 = 0.;

        // Convergence iterations
        for i in 0..50 {
            // Get predicted state and apply to element
            let Q: Matrix7xX = state_next.Q();
            let V: Matrix6xX = state_next.V();
            let A: Matrix6xX = state_next.A();
            elem.update_states(&Q, &V, &A, &self.gravity);

            // Get element residual vector, stiffness, damping, and mass matrices
            let R_FE: VectorD = elem.R_FE();
            let K_FE: MatrixD = elem.K_FE();
            let M: MatrixD = elem.M();
            let G: MatrixD = elem.G();

            // Constraint contribution to static iteration matrix
            let K_C: MatrixD = MatrixD::zeros(K_FE.nrows(), K_FE.ncols());

            // Tangent operator
            let T: MatrixD = state_next.tangent_operator();

            // Assemble the residual vector
            self.R.fill(0.);
            self.R.rows_mut(0, num_node_dofs).add_assign(&(R_FE));

            // Assemble iteration matrix
            self.St.fill(0.);
            // Quadrant 1,1
            self.St
                .view_mut((0, 0), (num_node_dofs, num_node_dofs))
                .add_assign(
                    M * state_next.cfg.beta_prime
                        + G * state_next.cfg.gamma_prime
                        + (K_FE + K_C) * &T,
                );

            // If there are constraint nodes
            if self.num_constraint_nodes > 0 {
                // Get constraints_gradient_matrix
                let B: Matrix6xX = elem.constraints_gradient_matrix();

                // Get constraints residual vector
                let F_C: VectorD = B.transpose() * &lambda;

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

            let mut x = VectorD::zeros(num_node_dofs);
            // Solve system
            if num_constraint_dofs == 0 {
                let xx: VectorD = St
                    .view((6, 6), (num_node_dofs - 6, num_node_dofs - 6))
                    .lu()
                    .solve(&R.rows(6, num_node_dofs - 6))
                    .expect("Matrix is not invertable");
                x.rows_mut(6, num_node_dofs - 6).copy_from(&xx);
            } else {
                x = St.lu().solve(&R).expect("Matrix is not invertable");
            }

            // Un-condition the solution vector
            let x: VectorD = -x.component_mul(&self.DR_diag);

            // Extract delta x and delta lambda from system solution
            let delta_x: Matrix6xX =
                Matrix6xX::from_column_slice(x.rows(0, num_node_dofs).as_slice());
            let lambda_delta: VectorD = x.rows(num_node_dofs, num_constraint_dofs).clone_owned();

            // update state and lambda
            state_next.update(&delta_x);
            lambda += lambda_delta;

            // Check for convergence
            let energy_increment = self.R.dot(&x).abs();
            res_norm.push(energy_increment);
            if i == 0 {
                energy_increment_ref = energy_increment;
            }
            if energy_increment < 1e-8 || (energy_increment / energy_increment_ref < 1e-5) {
                state_next.update_algorithmic_acceleration();
                self.state = state_next;
                return Some(res_norm);
            }
        }
        // Solution did not converge
        None
    }
}
