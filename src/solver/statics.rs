#![allow(non_snake_case)]

use std::ops::AddAssign;

use crate::element::{
    gebt::{Nodes, QuatExt},
    interp::{
        gauss_legendre_lobotto_points, lagrange_polynomial_derivative,
        quaternion_from_tangent_twist,
    },
};
use nalgebra::{
    DMatrix, DVector, Dyn, Matrix6xX, OMatrix, Quaternion, UnitQuaternion, Vector4, U7,
};

type Matrix7xX = OMatrix<f64, U7, Dyn>;

#[derive(Debug, Clone)]
pub struct State {
    q: Matrix7xX,       // displacement
    v: Matrix6xX<f64>,  // velocity
    vd: Matrix6xX<f64>, // acceleration
    a: Matrix6xX<f64>,  // algorithm acceleration
}

impl State {
    pub fn new(num_nodes: usize) -> Self {
        State {
            q: Matrix7xX::zeros(num_nodes),
            v: Matrix6xX::zeros(num_nodes),
            vd: Matrix6xX::zeros(num_nodes),
            a: Matrix6xX::zeros(num_nodes),
        }
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
    state: State,
    state_next: State,
    dq: Matrix6xX<f64>,     // delta displacement+rotation
    lambda: Matrix6xX<f64>, // constraints
    St: DMatrix<f64>,
    R: DVector<f64>,
    ndofs: usize,
}

impl GeneralizedAlphaSolver {
    pub fn new(state0: State, num_constraint_nodes: usize, rho_inf: f64, t0: f64, h: f64) -> Self {
        // Generalized alpha parameters
        let alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
        let alpha_f = rho_inf / (rho_inf + 1.);
        let gamma = 0.5 + alpha_f - alpha_m;
        let beta = 0.25 * (gamma + 0.5).powi(2);
        let beta_prime = (1. - alpha_m) / (h * h * beta * (1. - alpha_f));
        let gamma_prime = gamma / (h * beta);

        // Number of DOFs in full system with constraints
        let ndofs = (state0.q.ncols() + num_constraint_nodes) * 6;

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
            state: state0.clone(),
            state_next: state0.clone(),
            dq: Matrix6xX::zeros(state0.q.ncols()),
            lambda: Matrix6xX::zeros(num_constraint_nodes),
            St: DMatrix::zeros(ndofs, ndofs),
            R: DVector::zeros(ndofs),
            ndofs,
        }
    }

    pub fn predict_next_state(&mut self) {
        // Initialize acceleration to zero
        self.state_next.vd.fill(0.0);

        // Initialize lambda
        self.lambda.fill(0.0);

        // Calculate new algorithmic acceleration
        self.state_next.a =
            (self.alpha_f * &self.state.vd - self.alpha_m * &self.state.a) / (1. - self.alpha_m);

        // Calculate new velocity
        self.state_next.v = &self.state.v
            + self.h * (1. - self.gamma) * &self.state.a
            + self.gamma * self.h * &self.state_next.a;

        // Calculate change in displacement/rotation (average velocity)
        self.dq = &self.state.v
            + (0.5 - self.beta) * self.h * &self.state.a
            + self.beta * self.h * &self.state_next.a;
    }

    pub fn update_next_displacement(&mut self) {
        // Update displacements
        let mut u = self.state_next.q.fixed_rows_mut::<3>(0);
        let du = self.dq.fixed_rows::<3>(0);
        u.add_assign(du * self.h);

        // Update rotations
        let mut r = self.state_next.q.fixed_rows_mut::<4>(3);
        let dr = self.dq.fixed_rows::<3>(3);
        for (mut r, dr) in r.column_iter_mut().zip(dr.column_iter()) {
            // Get current rotation as a quaternion
            let R = UnitQuaternion::from_quaternion(Quaternion::new(r[0], r[1], r[2], r[3]));

            // Get change in rotation as quaternion
            let dR = UnitQuaternion::from_scaled_axis(self.h * dr);

            // Compose rotation with delta rotation to get new rotation
            let Rn = R * dR;

            // Update rotation vector
            r.copy_from(&Vector4::new(Rn.w, Rn.i, Rn.j, Rn.k));
        }
    }

    pub fn advance_state(&mut self) {
        self.state = self.state_next.clone()
    }
}

#[cfg(test)]
mod tests {

    use std::ops::AddAssign;

    use approx::assert_relative_eq;
    use itertools::Itertools;
    use nalgebra::{
        DMatrix, DVector, Matrix3, Matrix3xX, Matrix4xX, Matrix6, Matrix6xX, UnitQuaternion,
        Vector3, Vector4, Vector6,
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
        solver::statics::{GeneralizedAlphaSolver, State},
    };

    #[test]
    fn test_beam3() {
        //----------------------------------------------------------------------
        // Initial configuration
        //----------------------------------------------------------------------

        let fx = |s: f64| -> f64 { 10. * s + 1. };
        let fz = |s: f64| -> f64 { 0. };
        let fy = |s: f64| -> f64 { 0. };
        let ft = |s: f64| -> f64 { 0. };
        let xi: DVector<f64> = DVector::from_vec(gauss_legendre_lobotto_points(4));
        let s: DVector<f64> = xi.add_scalar(1.) / 2.;

        let num_nodes = s.len();

        // Node initial position
        let x0: Matrix3xX<f64> = Matrix3xX::from_iterator(
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
        let r: Matrix4xX<f64> = Matrix4xX::from_columns(
            &s.iter()
                .map(|&si| UnitQuaternion::from_matrix(&rot(si)).to_vector())
                .collect_vec(),
        );

        //----------------------------------------------------------------------
        // Do the stuff
        //----------------------------------------------------------------------

        // Quadrature rule
        let gq = Quadrature::gauss(7);

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

        // Create element from nodes
        let mut elem = nodes.element(&gq, &sections);

        // Get the deformed element
        elem.displace(&u, &r);

        // Create initial state
        let mut state0 = State::new(elem.nodes.num);
        state0.q.fixed_rows_mut::<3>(0).copy_from(&elem.nodes.u);
        state0.q.fixed_rows_mut::<4>(3).copy_from(&elem.nodes.r);

        // Create generalized alpha solver
        let rho_inf = 1.0;
        let t0 = 0.0;
        let h = 0.0;
        let mut solver = GeneralizedAlphaSolver::new(state0, 1, rho_inf, t0, h);

        // Get constraints_gradient_matrix
        let B: Matrix6xX<f64> = elem.constraints_gradient_matrix();
        assert_relative_eq!(B.columns(0, 6), Matrix6::identity().columns(0, 6));

        // Get residual vector
        let R: DVector<f64> = elem.residual_vector();
        assert_relative_eq!(
            R,
            DVector::from_vec(vec![
                0., 0.8856, 0., 0., 0., 4.428, 0., -3.01898, 0., 0., 0., -12.4884, 0., 9.4464, 0.,
                0., 0., 23.616, 0., -69.305, 0., 0., 0., -59.8356, 0., 61.992, 0., 0., 0., -44.28
            ]),
            epsilon = 1e-4
        );

        // Get residual constraints vector
        let R_C = B.transpose() * solver.lambda;

        // Get constraint_residual_vector
        let Phi: Vector6<f64> = elem.constraint_residual_vector();
        assert_relative_eq!(Phi, Vector6::zeros());

        // Static iteration matrix
        let K_FE: DMatrix<f64> = elem.stiffness_matrix();
        assert_relative_eq!(K_FE[(0, 0)], 957719.0, epsilon = 1e-5);
        assert_relative_eq!(K_FE[(1, 1)], 61992.0, epsilon = 1e-5);

        // Constraint contribution to static iteration matrix
        let K_C: DMatrix<f64> = DMatrix::zeros(K_FE.nrows(), K_FE.ncols());

        // Number of degrees of freedom
        let num_constraint_dofs = B.nrows();
        let num_node_dofs = K_FE.nrows();

        // Assemble iteration matrix
        solver.St.fill(0.);
        solver
            .St
            .view_mut((0, 0), (num_node_dofs, num_node_dofs))
            .add_assign(&(K_FE + K_C));
        solver
            .St
            .view_mut((num_node_dofs, 0), (num_constraint_dofs, num_node_dofs))
            .add_assign(&B);
        solver
            .St
            .view_mut((0, num_node_dofs), (num_node_dofs, num_constraint_dofs))
            .add_assign(&B.transpose());

        // Assemble the residual vector
        solver.R.fill(0.);
        solver.R.rows_mut(0, num_node_dofs).add_assign(&(R + R_C));
        solver
            .R
            .rows_mut(num_node_dofs, num_constraint_dofs)
            .add_assign(Phi);

        let x = solver
            .St
            .lu()
            .solve(&solver.R)
            .expect("Matrix is not invertable");

        print!("{}", x)
    }
}
