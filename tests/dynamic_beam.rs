#![allow(non_snake_case)]

use std::io::Write;

use roturb::{
    element::{
        gebt::{Material, Nodes, Section},
        interp::{
            gauss_legendre_lobotto_points, lagrange_polynomial_derivative,
            quaternion_from_tangent_twist,
        },
        quadrature::Quadrature,
    },
    prelude::*,
    solver::{GeneralizedAlphaSolver, State, StepConfig},
};

#[test]
fn test_cantilever_beam_with_with_sin_load() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let t0: f64 = 0.;
    let tf: f64 = 4.;
    let h: f64 = 0.005;

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

    // Create material
    let mat = Material {
        M_star: Matrix6::from_row_slice(&vec![
            8.538, 0.000, 0.000, 0.000, 0.000, 0.000, // Row 6
            0.000, 8.538, 0.000, 0.000, 0.000, 0.000, // Row 5
            0.000, 0.000, 8.538, 0.000, 0.000, 0.000, // Row 4
            0.000, 0.000, 0.000, 1.4433, 0.000, 0.000, // Row 3
            0.000, 0.000, 0.000, 0.000, 0.40972, 0.000, // Row 2
            0.000, 0.000, 0.000, 0.000, 0.000, 1.0336, // Row 1
        ]) * 1e-2,
        C_star: Matrix6::from_row_slice(&vec![
            1368.17, 0., 0., 0., 0., 0., // Row 1
            0., 88.56, 0., 0., 0., 0., // Row 2
            0., 0., 38.78, 0., 0., 0., // Row 3
            0., 0., 0., 16.960, 17.610, -0.351, // Row 4
            0., 0., 0., 17.610, 59.120, -0.370, // Row 5
            0., 0., 0., -0.351, -0.370, 141.47, // Row 6
        ]) * 1e3,
    };

    // Create sections
    let sections: Vec<Section> = vec![Section::new(0.0, &mat), Section::new(1.0, &mat)];

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    // Create nodes structure
    let nodes = Nodes::new(&s, &xi, &x0, &r0);

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Create element from nodes
    let mut elem = nodes.element(&gq, &sections);

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Number of state and constraint nodes
    let num_state_nodes = elem.nodes.num;
    let num_constraint_nodes = 1;

    // Create generalized alpha step configuration
    let step_config: StepConfig = StepConfig::new(rho_inf, h);

    // Create initial state
    let state0: State = State::new(&step_config, num_state_nodes, t0);

    // Create generalized alpha solver
    let mut solver =
        GeneralizedAlphaSolver::new(num_state_nodes, num_constraint_nodes, &state0, gravity);

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 1.0e2 * (10.0 * t).sin() };

    let mut file = std::fs::File::create("q_cbc.csv").expect("file failure");

    while solver.state.time() <= tf {
        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = point_force(solver.state.time() + h);
        elem.apply_force(&forces);

        // Solve time step
        match solver.step(&mut elem) {
            None => {
                print!("{:.3}s: failed to converge \n", solver.state.time() + h);
                break;
            }
            Some(energy_incs) => {
                print!(
                    "{:.3}s: converged in {} iterations\n",
                    solver.state.time(),
                    energy_incs.len()
                );
                states.push(solver.state.clone());
                file.write_fmt(format_args!("{:?}", solver.state.time()))
                    .expect("fail");
                file.write_fmt(format_args!(",{:?}", energy_incs.len()))
                    .expect("fail");
                for &v in solver.state.Q().iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");
            }
        }
    }

    // Assert that solver finished simulation
    // assert_relative_eq!(solver.state.t, tf, epsilon = 1e-8);
}

#[test]
fn test_cantilever_beam_with_with_sin_load_dirichlet_bc() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let t0: f64 = 0.;
    let tf: f64 = 4.;
    let h: f64 = 0.005;

    let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(4));
    let s: VectorN = xi.add_scalar(1.) / 2.;
    let num_nodes = s.len();

    // Node initial position and rotation
    let fx = |s: f64| -> f64 { 10. * s };
    let mut x0: Matrix3xX = Matrix3xX::zeros(num_nodes);
    for (mut c, &si) in x0.column_iter_mut().zip(s.iter()) {
        c[0] = fx(si);
    }
    let mut r0: Matrix4xX = Matrix4xX::zeros(num_nodes);
    r0.fixed_rows_mut::<1>(0).fill(1.);

    // Create material
    let mat = Material {
        M_star: Matrix6::from_row_slice(&vec![
            8.538, 0.000, 0.000, 0.000, 0.000, 0.000, // Row 6
            0.000, 8.538, 0.000, 0.000, 0.000, 0.000, // Row 5
            0.000, 0.000, 8.538, 0.000, 0.000, 0.000, // Row 4
            0.000, 0.000, 0.000, 1.4433, 0.000, 0.000, // Row 3
            0.000, 0.000, 0.000, 0.000, 0.40972, 0.000, // Row 2
            0.000, 0.000, 0.000, 0.000, 0.000, 1.0336, // Row 1
        ]) * 1e-2,
        C_star: Matrix6::from_row_slice(&vec![
            1368.17, 0., 0., 0., 0., 0., // Row 1
            0., 88.56, 0., 0., 0., 0., // Row 2
            0., 0., 38.78, 0., 0., 0., // Row 3
            0., 0., 0., 16.960, 17.610, -0.351, // Row 4
            0., 0., 0., 17.610, 59.120, -0.370, // Row 5
            0., 0., 0., -0.351, -0.370, 141.47, // Row 6
        ]) * 1e3,
    };

    // Create sections
    let sections: Vec<Section> = vec![Section::new(0.0, &mat), Section::new(1.0, &mat)];

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    // Create nodes structure
    let nodes = Nodes::new(&s, &xi, &x0, &r0);

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Create element from nodes
    let mut elem = nodes.element(&gq, &sections);

    //--------------------------------------------------------------------------
    // Apply loads
    //--------------------------------------------------------------------------

    // Number of state and constraint nodes
    let num_state_nodes = elem.nodes.num;
    let num_constraint_nodes = 0;

    // Create generalized alpha step configuration
    let step_config: StepConfig = StepConfig::new(rho_inf, h);

    // Create initial state
    let state0: State = State::new(&step_config, num_state_nodes, t0);

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Create generalized alpha solver
    let mut solver =
        GeneralizedAlphaSolver::new(num_state_nodes, num_constraint_nodes, &state0, gravity);

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 1.0e2 * (10.0 * t).sin() };

    let mut file = std::fs::File::create("q_dbc.csv").expect("file failure");

    while solver.state.time() <= tf {
        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = point_force(solver.state.time() + h);
        elem.apply_force(&forces);

        // Solve time step
        match solver.step(&mut elem) {
            None => {
                print!("{:.3}s: failed to converge \n", solver.state.time() + h);
                break;
            }
            Some(energy_incs) => {
                print!(
                    "{:.3}s: converged in {} iterations\n",
                    solver.state.time(),
                    energy_incs.len()
                );
                states.push(solver.state.clone());
                file.write_fmt(format_args!("{:?}", solver.state.time()))
                    .expect("fail");
                file.write_fmt(format_args!(",{:?}", energy_incs.len()))
                    .expect("fail");
                for &v in solver.state.Q().columns(num_nodes - 1, 1).iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");
            }
        }
    }
}
