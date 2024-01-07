#![allow(non_snake_case)]

use std::ops::Sub;

use approx::assert_relative_eq;
use roturb::{
    element::{
        gebt::{Element, Material, Nodes, Section},
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
fn test_static_element() {
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
        s.iter()
            .zip(tangent.column_iter())
            .map(|(&si, tan)| quaternion_from_tangent_twist(&Vector3::from(tan), ft(si)).wijk())
            .collect::<Vec<Vector4>>()
            .as_slice(),
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

    //--------------------------------------------------------------------------
    // Step configuration
    //--------------------------------------------------------------------------

    let step_config = StepConfig::new(1.0, 1.0);

    let gravity: Vector3 = Vector3::zeros();

    let is_dynamic_solve = false;

    //--------------------------------------------------------------------------
    // Test solve of element with no initial displacement
    //--------------------------------------------------------------------------

    // Create initial state
    let state0: State = State::new(&step_config, elem.nodes.num, 0.);

    // Create generalized alpha solver
    let mut solver =
        GeneralizedAlphaSolver::new(elem.nodes.num, 1, &state0, gravity, is_dynamic_solve);

    // Solve time step
    let errors = solver.step(&mut elem).expect("solution failed to converge");
    assert_eq!(errors.len(), 1);

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
    let state0: State = State::new_with_initial_state(&step_config, elem.nodes.num, 0., &Q, &V, &A);

    // Create generalized alpha solver
    let mut solver =
        GeneralizedAlphaSolver::new(elem.nodes.num, 1, &state0, gravity, is_dynamic_solve);

    // Solve time step
    let errors = solver.step(&mut elem).expect("solution failed to converge");
    assert_eq!(errors.len(), 2);

    //--------------------------------------------------------------------------
    // Test solve of element with applied load
    //--------------------------------------------------------------------------

    // Create initial state
    let state0: State = State::new(&step_config, elem.nodes.num, 0.);

    // Create generalized alpha solver
    let mut solver =
        GeneralizedAlphaSolver::new(elem.nodes.num, 1, &state0, gravity, is_dynamic_solve);

    // Create force matrix, apply 150 lbs force in z direction of last node
    let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);
    forces[(2, num_nodes - 1)] = 150.;
    elem.apply_force(&forces);

    let errors = solver.step(&mut elem).expect("solution failed to converge");

    // Verify number of convergence iterations
    assert_eq!(errors.len(), 5);

    // Verify end node displacement in xyz
    let q = solver.state.q();
    assert_relative_eq!(
        Vector3::from(q.fixed_view::<3, 1>(0, num_nodes - 1)),
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
        Matrix4xX::from_column_slice(&s.iter().flat_map(|&si| vec![1., 0., 0., 0.]).collect_vec());

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

    let step_config = StepConfig::new(1.0, 1.0);
    let gravity: Vector3 = Vector3::zeros();

    //--------------------------------------------------------------------------
    // Test solve of element with applied load
    //--------------------------------------------------------------------------

    // Create initial state
    let state0: State = State::new(&step_config, elem.nodes.num, 0.);

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        elem.nodes.num,
        num_constraint_nodes,
        &state0,
        gravity,
        is_dynamic_solve,
    );

    // Create force matrix, apply 150 lbs force in z direction of last node
    let mut forces: Matrix6xX = Matrix6xX::zeros(num_nodes);

    // Create empty iter directory
    let _ = std::fs::remove_dir_all("iter");
    std::fs::create_dir("iter").unwrap();

    // Loop through moments
    for (i, &m) in Mz.iter().enumerate() {
        forces[(4, num_nodes - 1)] = -m;
        elem.apply_force(&forces);
        let iter_data = solver.step(&mut elem).expect("solution failed to converge");
        println!("Mz={:6}, niter={}", m, iter_data.len());
        let vtk = element_vtk(&elem);
        vtk.export_ascii(format!("iter/step_{:0>3}.vtk", i))
            .unwrap();
    }
}

use vtkio::model::*; // import model definition of a VTK file

fn element_vtk(elem: &Element) -> Vtk {
    let rotations: Vec<Matrix3> = elem
        .nodes
        .r0
        .column_iter()
        .zip(elem.nodes.r.column_iter())
        .map(|(r0, r)| {
            (r0.clone_owned().as_unit_quaternion() * r.clone_owned().as_unit_quaternion())
                .to_rotation_matrix()
                .matrix()
                .clone_owned()
        })
        .collect_vec();
    let orientations = vec!["OrientationX", "OrientationY", "OrientationZ"];

    Vtk {
        version: Version { major: 4, minor: 2 },
        title: String::new(),
        byte_order: ByteOrder::LittleEndian,
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64((&elem.nodes.u + &elem.nodes.x0).as_slice().to_vec()),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity: {
                        let mut a = vec![0, elem.nodes.num - 1];
                        let b = (1..elem.nodes.num - 1).collect_vec();
                        a.extend(b);
                        a.iter().map(|&i| i as u64).collect_vec()
                    },
                    offsets: vec![elem.nodes.num as u64],
                },
                types: vec![CellType::LagrangeCurve],
            },
            data: Attributes {
                point: orientations
                    .iter()
                    .enumerate()
                    .map(|(i, &orientation)| {
                        Attribute::DataArray(DataArrayBase {
                            name: orientation.to_string(),
                            elem: ElementType::Vectors,
                            data: IOBuffer::F32(
                                rotations
                                    .iter()
                                    .flat_map(|r| {
                                        r.column(i)
                                            .iter()
                                            .map(|&v| ((v * 1.0e7).round() / 1.0e7) as f32)
                                            .collect_vec()
                                    })
                                    .collect_vec(),
                            ),
                        })
                    })
                    .collect_vec(),
                ..Default::default()
            },
        }),
    }
}
