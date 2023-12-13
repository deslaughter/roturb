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
    solver::{GeneralizedAlphaSolver, State},
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

    // Write displacements to csv
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
fn test_cantilever_beam_with_with_sin_load() {
    // Solver parameters
    let rho_inf: f64 = 0.;
    let t0: f64 = 0.;
    let h: f64 = 0.05;

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
        mass: Matrix6::from_row_slice(&vec![
            8.538, 0.000, 0.000, 0.000, 0.000, 0.000, // Row 6
            0.000, 8.538, 0.000, 0.000, 0.000, 0.000, // Row 5
            0.000, 0.000, 8.538, 0.000, 0.000, 0.000, // Row 4
            0.000, 0.000, 0.000, 1.4433, 0.000, 0.000, // Row 3
            0.000, 0.000, 0.000, 0.000, 0.40972, 0.000, // Row 2
            0.000, 0.000, 0.000, 0.000, 0.000, 1.0336, // Row 1
        ]) * 1e-2,
        // mass: Matrix6::zeros(),
        stiffness: Matrix6::from_row_slice(&vec![
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

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Number of state and constraint nodes
    let num_state_nodes = elem.nodes.num;
    let num_constraint_nodes = 1;

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_state_nodes,
        num_constraint_nodes,
        rho_inf,
        t0,
        h,
        gravity,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 1.0e2 * (10.0 * t).sin() };

    while solver.state.t <= 4.0 {
        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
        forces[(2, num_nodes - 1)] = point_force(solver.state.t + h);
        elem.apply_force(&forces);

        // Solve time step
        match solver.solve_time_step(&mut elem) {
            None => {
                print!("{:?}s: failed to converge \n", solver.state.t + solver.h);
                break;
            }
            Some(niter) => {
                print!("{:?}s: converged in {} iterations\n", solver.state.t, niter);
                states.push(solver.state.clone());
            }
        }
    }

    // Write displacements to csv
    let mut file = std::fs::File::create("q.csv").expect("file failure");
    for s in states {
        file.write_fmt(format_args!("{:?}", s.t)).expect("fail");
        for v in s.q.iter() {
            file.write_fmt(format_args!(",{:?}", v)).expect("fail");
        }
        file.write_all(b"\n").expect("fail");
    }
}
