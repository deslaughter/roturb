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
    solver::GeneralizedAlphaSolver,
};

#[test]
fn test_static_element() {
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
        mass: Matrix6::zeros(),
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

    // Verify number of convergence iterations
    assert_eq!(num_conv_iter, 7);

    // Verify end node displacement in xyz
    assert_relative_eq!(
        Vector3::from(solver.state.q.fixed_view::<3, 1>(0, num_nodes - 1)),
        Vector3::new(
            -0.09020517194956187,
            -0.06451290662897675,
            -1.2287954559156433
        ),
        epsilon = 1e-6
    )
}
