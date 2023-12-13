#![allow(non_snake_case)]

use crate::prelude::*;

use crate::element::gebt::Element;

#[derive(Debug, Clone)]
pub struct State {
    t: f64,             // time
    q: Matrix7xX,       // displacement
    q_delta: Matrix6xX, // change in displacement
    v: Matrix6xX,       // velocity
    vd: Matrix6xX,      // acceleration
    a: Matrix6xX,       // algorithm acceleration
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

//------------------------------------------------------------------------------
// Testing
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use std::io::Write;

    use super::*;

    use crate::{
        element::{
            gebt::{Material, Nodes, Section},
            interp::{
                gauss_legendre_lobotto_points, lagrange_polynomial_derivative,
                quaternion_from_tangent_twist,
            },
            quadrature::Quadrature,
        },
        solver::GeneralizedAlphaSolver,
    };

    #[test]
    fn test_dynamic_element_with_point_load() {
        //----------------------------------------------------------------------
        // Initial configuration
        //----------------------------------------------------------------------

        let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(4));
        let s: VectorN = xi.add_scalar(1.) / 2.;
        let num_nodes = s.len();

        let fx = |s: f64| -> f64 { 10. * s };
        let fz = |s: f64| -> f64 { 0. * s };
        let fy = |s: f64| -> f64 { 0. * s };
        let ft = |s: f64| -> f64 { 0. * s };

        // Node initial position
        let x0: Matrix3xX = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![fx(si), fy(si), fz(si)]),
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

        //----------------------------------------------------------------------
        // Material
        //----------------------------------------------------------------------

        // Cross section properties
        let t: f64 = 0.03; // wall thickness (in)
        let b: f64 = 0.953; // width (in)
        let bi = b - 2. * t;
        let h: f64 = 0.531; // height (in)
        let hi = h - 2. * t;
        let A = b * h - bi * hi; // Area (in^2)
        let Ixx = (b * h.powi(3) - bi * hi.powi(3)) / 12.; // (in^4)
        let Iyy = (h * b.powi(3) - hi * bi.powi(3)) / 12.; // (in^4)
        let Izz = Ixx + Iyy; // (in^4)
        let m: f64 = A * 0.1 / 386.4; // linear mass aluminum (lbm-s^2/in^2)

        // Create material
        let mat = Material {
            mass: Matrix6::from_diagonal(&Vector6::new(m, m, m, Ixx, Iyy, Izz)),
            // mass: Matrix6::zeros(),
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
        // Apply loads
        //----------------------------------------------------------------------

        // Apply force
        let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = 1.;
        elem.apply_force(&forces);

        //----------------------------------------------------------------------
        // Test solve of element with initial displacement
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver =
            GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 0.1, Vector3::zeros());

        let mut states = vec![solver.state.clone()];

        for _ in 0..10 {
            // Solve time step
            let num_conv_iter = solver
                .solve_time_step(&mut elem)
                .expect("solution failed to converge");

            print!(
                "{}",
                solver.state.q.view((0, num_nodes - 1), (3, 1)).transpose()
            );
            states.push(solver.state.clone());
        }

        let mut file = std::fs::File::create("q.csv").expect("file failure");
        for s in states {
            file.write_fmt(format_args!("{:?}", s.t)).expect("fail");
            for v in s.q.iter() {
                file.write_fmt(format_args!(",{:?}", v)).expect("fail");
            }
            file.write_all(b"\n").expect("fail");
        }
    }

    #[test]
    fn test_static_element_initial_disp() {
        //----------------------------------------------------------------------
        // Initial configuration
        //----------------------------------------------------------------------

        let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(4));
        let s: VectorN = xi.add_scalar(1.) / 2.;
        let num_nodes = s.len();

        let fx = |s: f64| -> f64 { 10. * s };
        let fz = |s: f64| -> f64 { 0. * s };
        let fy = |s: f64| -> f64 { 0. * s };
        let ft = |s: f64| -> f64 { 0. * s };

        // Node initial position
        let x0: Matrix3xX = Matrix3xX::from_iterator(
            num_nodes,
            s.iter().flat_map(|&si| vec![fx(si), fy(si), fz(si)]),
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

        // Get matrix describing the nodal rotation
        let r: Matrix4xX = Matrix4xX::from_columns(
            &s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).wijk())
                .collect_vec(),
        );

        //----------------------------------------------------------------------
        // Material
        //----------------------------------------------------------------------

        // Create material
        let mat = Material {
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
        let mut solver =
            GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0, Vector3::zeros());

        // Solve time step
        let num_conv_iter = solver
            .solve_time_step(&mut elem)
            .expect("solution failed to converge");
        assert_eq!(num_conv_iter, 0);

        //----------------------------------------------------------------------
        // Test solve of element with initial displacement
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver =
            GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0, Vector3::zeros());

        // Populate initial state
        solver.state.q.fixed_rows_mut::<3>(0).copy_from(&u);
        solver.state.q.fixed_rows_mut::<4>(3).copy_from(&r);

        // Solve time step
        let num_conv_iter = solver
            .solve_time_step(&mut elem)
            .expect("solution failed to converge");
        assert_eq!(num_conv_iter, 1);

        //----------------------------------------------------------------------
        // Test solve of element with applied load
        //----------------------------------------------------------------------

        // Create generalized alpha solver
        let mut solver =
            GeneralizedAlphaSolver::new(elem.nodes.num, 1, 1.0, 0.0, 1.0, Vector3::zeros());

        // Create force matrix, apply 150 lbs force in z direction of last node
        let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = 150.;
        elem.apply_force(&forces);

        let num_conv_iter = solver
            .solve_time_step(&mut elem)
            .expect("solution failed to converge");
    }
}
