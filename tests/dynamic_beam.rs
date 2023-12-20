#![allow(non_snake_case)]

use std::{f64::consts::PI, io::Write};

use roturb::{
    element::{
        gebt::{Element, Material, Nodes, Section},
        interp::gauss_legendre_lobotto_points,
        quadrature::Quadrature,
    },
    prelude::*,
    solver::{GeneralizedAlphaSolver, State, StepConfig},
};

fn build_element() -> Element {
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
    // let gq = Quadrature::gauss_legendre_lobotto(5); // Nodal quadrature

    // Create element from nodes
    nodes.element(&gq, &sections)
}

#[test]
fn test_cantilever_beam_with_with_sin_load() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let t0: f64 = 0.;
    let tf: f64 = 4.;
    let h: f64 = 0.005;

    let mut elem = build_element();

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
    let mut solver = GeneralizedAlphaSolver::new(
        num_state_nodes,
        num_constraint_nodes,
        &state0,
        gravity,
        true,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 1.0e2 * (10.0 * t).sin() };

    let mut file = std::fs::File::create("q_cbc.csv").expect("file failure");

    while solver.state.time() <= tf {
        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        forces[(2, elem.nodes.num - 1)] = point_force(solver.state.time() + h);
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
                for &v in solver.state.q().iter() {
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

    let mut elem = build_element();

    //--------------------------------------------------------------------------
    // Create element and solver
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
    let mut solver = GeneralizedAlphaSolver::new(
        num_state_nodes,
        num_constraint_nodes,
        &state0,
        gravity,
        true,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 100. * (10. * t).sin() };

    let mut file = std::fs::File::create("q_dbc.csv").expect("file failure");

    while solver.state.time() <= tf {
        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        forces[(2, elem.nodes.num - 1)] = point_force(solver.state.time() + h);
        elem.apply_force(&forces);

        // Solve time step
        match solver.step(&mut elem) {
            None => {
                print!("{:.3}s: failed to converge \n", solver.state.time() + h);
                break;
            }
            Some(iter_data) => {
                print!(
                    "{:.3}s: converged in {} iterations\n",
                    solver.state.time(),
                    iter_data.len()
                );
                states.push(solver.state.clone());
                file.write_fmt(format_args!("{:?}", solver.state.time()))
                    .expect("fail");
                file.write_fmt(format_args!(",{:?}", iter_data.len()))
                    .expect("fail");
                for &v in solver.state.q().columns(elem.nodes.num - 1, 1).iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");
            }
        }
    }
}

#[test]
fn test_rotating_beam() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let t0: f64 = 0.;
    let tf: f64 = 20.;
    let h: f64 = 0.005;

    let mut elem = build_element();

    // let rot0 = PI;
    let rot0 = 0.;
    let r0: UnitQuaternion = UnitQuaternion::from_euler_angles(0., 0., rot0);

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Number of state and constraint nodes
    let num_state_nodes = elem.nodes.num;
    let num_constraint_nodes = 0;

    // Create generalized alpha step configuration
    let step_config: StepConfig = StepConfig::new(rho_inf, h);

    // Create initial state
    let mut Q: Matrix7xX = Matrix7xX::zeros(elem.nodes.num);
    for (i, mut c) in Q.column_iter_mut().enumerate() {
        c.rows_mut(0, 3)
            .copy_from(&(r0 * elem.nodes.x0.column(i) - elem.nodes.x0.column(i)));
        c.rows_mut(3, 4).copy_from(&r0.wijk());
    }
    let V: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
    let A: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
    let state0: State =
        State::new_with_initial_state(&step_config, num_state_nodes, t0, &Q, &V, &A);

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_state_nodes,
        num_constraint_nodes,
        &state0,
        gravity,
        true,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![solver.state.clone()];

    let mut file = std::fs::File::create("q_rot.csv").expect("file failure");
    let _ = std::fs::remove_dir_all("vtk");
    std::fs::create_dir("vtk").unwrap();
    let _ = std::fs::remove_dir_all("iter");
    std::fs::create_dir("iter").unwrap();

    let num_steps = (tf / h).ceil() as usize + 1;

    for i in 0..num_steps {
        let t = (i as f64) * h;

        // Prescribe element root displacement
        let rotz = 0.5 * (t + h) + rot0;
        let r: UnitQuaternion = UnitQuaternion::from_euler_angles(0., 0., rotz);
        elem.q_root.fixed_rows_mut::<4>(3).copy_from(&r.wijk());

        // Replace solver state
        // let rotz = 0.5 * t + rot0;
        // let r: UnitQuaternion = UnitQuaternion::from_euler_angles(0., 0., rotz);
        // let mut Q: Matrix7xX = Matrix7xX::zeros(elem.nodes.num);
        // for (i, mut c) in Q.column_iter_mut().enumerate() {
        //     c.rows_mut(0, 3)
        //         .copy_from(&(r * elem.nodes.x0.column(i) - elem.nodes.x0.column(i)));
        //     c.rows_mut(3, 4).copy_from(&r.wijk());
        // }
        // let V: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        // let A: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        // solver.state = State::new_with_initial_state(&step_config, num_state_nodes, t, &Q, &V, &A);

        // Solve time step
        match solver.step(&mut elem) {
            None => {
                print!("{:.3}s: failed to converge \n", solver.state.time() + h);
                break;
            }
            Some(iter_data) => {
                let t = solver.state.time();
                print!(
                    "Step {:3}, t={:.3}s: converged in {:2} iterations ({:.5} rad)\n",
                    i,
                    t,
                    iter_data.len(),
                    rotz
                );
                states.push(solver.state.clone());
                file.write_fmt(format_args!("{:?}", t)).expect("fail");
                file.write_fmt(format_args!(",{:?}", iter_data.len()))
                    .expect("fail");
                for &v in solver.state.q().iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");

                let vtk = element_vtk(&elem);
                vtk.export_ascii(format!("vtk/step_{:0>3}.vtk", i)).unwrap();

                let mut f =
                    std::fs::File::create(format!("iter/step_{:0>3}_conv_x.csv", i)).unwrap();
                for (i, idata) in iter_data.iter().enumerate() {
                    f.write_fmt(format_args!("{:?}", i + 1)).expect("fail");
                    for &v in idata.x.iter() {
                        f.write_fmt(format_args!(",{:?}", v)).expect("fail");
                    }
                    f.write_all(b"\n").expect("fail");
                }
            }
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
}
