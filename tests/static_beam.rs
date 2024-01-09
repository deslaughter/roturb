#![allow(non_snake_case)]

use approx::assert_relative_eq;
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
fn test_static_element() {
    let rho_inf = 1.0;
    let h = 1.0;
    let num_constraint_nodes = 1;

    //--------------------------------------------------------------------------
    // Initial configuration
    //--------------------------------------------------------------------------

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
        &s.iter()
            .zip(tangent.column_iter())
            .map(|(&si, tan)| quaternion_from_tangent_twist(&Vector3::from(tan), ft(si)).wijk())
            .collect_vec(),
    );

    //--------------------------------------------------------------------------
    // Displacements and rotations from reference
    //--------------------------------------------------------------------------

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
    let rot = |t: f64| -> UnitQuaternion { UnitQuaternion::from_euler_angles(scale * t, 0., 0.) };

    // Get xyz displacements at node locations
    let u: Matrix3xX = Matrix3xX::from_iterator(
        num_nodes,
        s.iter().flat_map(|&si| vec![ux(si), uy(si), uz(si)]),
    );

    // Get matrix describing the nodal rotation
    let r: Matrix4xX = Matrix4xX::from_columns(&s.iter().map(|&si| rot(si).wijk()).collect_vec());

    //--------------------------------------------------------------------------
    // Material
    //--------------------------------------------------------------------------

    // Create material
    let mat = Material {
        M_star: Matrix6::zeros(),
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

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    // Create nodes structure
    let nodes = Nodes::new(&s, &xi, &x0, &r0);

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Create element from nodes
    let mut elem = nodes.element(&gq, &sections);

    let num_system_nodes = elem.nodes.num;

    //--------------------------------------------------------------------------
    // Step configuration
    //--------------------------------------------------------------------------

    let gravity: Vector3 = Vector3::zeros();

    let is_dynamic_solve = false;

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_system_nodes,
        num_constraint_nodes,
        rho_inf,
        h,
        gravity,
        is_dynamic_solve,
    );

    //--------------------------------------------------------------------------
    // Test solve of element with no initial displacement
    //--------------------------------------------------------------------------

    // Create initial state
    let state: State = State::new(num_system_nodes, num_constraint_nodes);

    // Solve time step
    let (_, iter_data) = solver
        .step(0, &mut elem, &state)
        .expect("solution failed to converge");
    assert_eq!(iter_data.len(), 1);

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Populate initial state
    let mut Q: Matrix7xX = Matrix7xX::zeros(u.ncols());
    Q.fixed_rows_mut::<3>(0).copy_from(&u);
    Q.fixed_rows_mut::<4>(3).copy_from(&r);
    let V: Matrix6xX = Matrix6xX::zeros(u.ncols());
    let A: Matrix6xX = Matrix6xX::zeros(u.ncols());

    // Create initial state
    let state: State =
        State::new_with_initial_state(num_system_nodes, num_constraint_nodes, &Q, &V, &A);

    // Solve time step
    let (state_next, _) = solver
        .step(0, &mut elem, &state)
        .expect("solution failed to converge");
    assert_relative_eq!(
        state_next
            .q
            .fixed_view::<3, 1>(0, state_next.q.ncols() - 1)
            .clone_owned(),
        Vector3::zeros()
    );

    //--------------------------------------------------------------------------
    // Test solve of element with applied load
    //--------------------------------------------------------------------------

    // Create initial state
    let state: State = State::new(num_system_nodes, num_constraint_nodes);

    // Create force matrix, apply 150 lbs force in z direction of last node
    let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
    forces[(2, num_nodes - 1)] = 150.;
    elem.apply_force(&forces);

    let (state_next, _) = solver
        .step(0, &mut elem, &state)
        .expect("solution failed to converge");

    // Verify end node displacement in xyz
    assert_relative_eq!(
        Vector3::from(state_next.q.fixed_view::<3, 1>(0, num_system_nodes - 1)),
        Vector3::new(
            -0.09019953380447539,
            -0.06472124330431532,
            1.2287474434408805
        ),
        epsilon = 1e-6
    )
}

#[test]
fn test_static_beam_curl() {
    let num_constraint_nodes = 1;
    let is_dynamic_solve = false;
    let rho_inf = 1.0;
    let h = 1.0;

    // Create list of moments to apply to end of beam
    let Mz = vec![0., 10920.0, 21840.0, 32761.0, 43681.0, 54601.0];

    //--------------------------------------------------------------------------
    // Initial configuration
    //--------------------------------------------------------------------------

    // Node locations
    let num_nodes = 21;
    let s: VectorN = VectorN::from_vec(
        (0..num_nodes)
            .into_iter()
            .map(|v| (v as f64) / ((num_nodes - 1) as f64))
            .collect_vec(),
    );
    let xi: VectorN = (2. * (s.add_scalar(-s.min())) / (s.max() - s.min())).add_scalar(-1.);

    // Quadrature rule
    let gq = Quadrature::gauss(30);
    // let gq = Quadrature::gauss_legendre_lobotto(10);

    // Node initial position
    let x0: Matrix3xX = Matrix3xX::from_column_slice(
        &s.iter()
            .flat_map(|&si| vec![10. * si, 0., 0.])
            .collect_vec(),
    );

    // Node initial rotation
    let r0: Matrix4xX =
        Matrix4xX::from_column_slice(&s.iter().flat_map(|&_si| vec![1., 0., 0., 0.]).collect_vec());

    //--------------------------------------------------------------------------
    // Material
    //--------------------------------------------------------------------------

    // Create material
    let mat = Material {
        M_star: Matrix6::zeros(),
        C_star: Matrix6::from_row_slice(&vec![
            1770.0e3, 0., 0., 0., 0., 0., // row 1
            0., 1770.0e3, 0., 0., 0., 0., // row 2
            0., 0., 1770.0e3, 0., 0., 0., // row 3
            0., 0., 0., 8.16e3, 0., 0., // row 4
            0., 0., 0., 0., 86.9e3, 0., // row 5
            0., 0., 0., 0., 0., 215.0e3, // row 6
        ]),
    };

    // Create sections
    let sections: Vec<Section> = vec![Section::new(0.0, &mat), Section::new(1.0, &mat)];

    //--------------------------------------------------------------------------
    // Create element
    //--------------------------------------------------------------------------

    // Create nodes structure
    let nodes = Nodes::new(&s, &xi, &x0, &r0);

    // Create element from nodes
    let mut elem = nodes.element(&gq, &sections);

    //--------------------------------------------------------------------------
    // Step configuration
    //--------------------------------------------------------------------------

    let gravity: Vector3 = Vector3::zeros();

    //--------------------------------------------------------------------------
    // Test solve of element with applied load
    //--------------------------------------------------------------------------

    // Create initial state
    let state: State = State::new(elem.nodes.num, num_constraint_nodes);

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        elem.nodes.num,
        num_constraint_nodes,
        rho_inf,
        h,
        gravity,
        is_dynamic_solve,
    );

    // Create force matrix, apply 150 lbs force in z direction of last node
    let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);

    // Create empty iter directory
    let _ = std::fs::remove_dir_all("iter");
    std::fs::create_dir("iter").unwrap();

    let mut u_tip: Matrix3xX = Matrix3xX::zeros(Mz.len());

    // Loop through moments
    for (i, &m) in Mz.iter().enumerate() {
        // Apply moment to tip node about y axis
        forces[(4, num_nodes - 1)] = -m;
        elem.apply_force(&forces);

        // Step
        let (state_next, iter_data) = solver
            .step(i, &mut elem, &state)
            .expect("solution failed to converge");
        println!("Mz={:6}, niter={}", m, iter_data.len());

        // Get tip displacement
        u_tip
            .column_mut(i)
            .copy_from(&state_next.q.fixed_view::<3, 1>(0, state_next.q.ncols() - 1));

        // Write VTK file of element
        elem.to_vtk()
            .export_ascii(format!("iter/step_{:0>3}.vtk", i))
            .unwrap();
    }

    assert_relative_eq!(
        u_tip,
        Matrix3xX::from_column_slice(&vec![
            0., 0., 0., // column 1
            -2.4317, 0., 5.4987, // column 2
            -7.6613, 0., 7.1978, // column 3
            -11.5591, 0., 4.7986, // column 4
            -11.8921, 0., 1.3747, // column 5
            -10.0000, 0., 0.0000, // column 6
        ]),
        epsilon = 1.0e-3
    )
}
