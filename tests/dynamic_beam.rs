#![allow(non_snake_case)]

use std::io::Write;

use roturb::{
    element::{
        gebt::{Element, Material, Nodes, Section},
        interp::gauss_legendre_lobotto_points,
        quadrature::Quadrature,
    },
    prelude::*,
    solver::{GeneralizedAlphaSolver, State},
};

fn build_element(R0: UnitQuaternion) -> Element {
    let xi: VectorN = VectorN::from_vec(gauss_legendre_lobotto_points(5));
    let s: VectorN = xi.add_scalar(1.) / 2.;
    let num_nodes = s.len();

    // Quadrature rule
    let gq = Quadrature::gauss(7);

    // Node initial position and rotation
    let fx = |s: f64| -> f64 { 10. * s + 2. };
    let mut x0: Matrix3xX = Matrix3xX::zeros(num_nodes);
    for (mut c, &si) in x0.column_iter_mut().zip(s.iter()) {
        c[0] = fx(si);
    }
    let x_root = x0.fixed_columns::<1>(0).clone_owned();
    for mut c in x0.column_iter_mut() {
        c.copy_from(&(R0 * (&c - x_root) + x_root));
    }
    let mut r0: Matrix4xX = Matrix4xX::zeros(num_nodes);
    for mut c in r0.column_iter_mut() {
        c.copy_from(&R0.wijk());
    }

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

    // Create element from nodes
    nodes.element(&gq, &sections)
}

#[test]
fn test_cantilever_beam_with_with_sin_load() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let tf: f64 = 4.;
    let h: f64 = 0.005;
    let is_dynamic_solve = true;

    let mut elem = build_element(UnitQuaternion::identity());

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Number of state and constraint nodes
    let num_system_nodes = elem.nodes.num;
    let num_constraint_nodes = 1;

    // Create initial state
    let mut state: State = State::new(num_system_nodes, num_constraint_nodes);

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_system_nodes,
        num_constraint_nodes,
        rho_inf,
        h,
        gravity,
        is_dynamic_solve,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 1.0e2 * (10.0 * t).sin() };

    let mut file = std::fs::File::create("q_cbc.csv").expect("file failure");

    // Loop through steps
    let num_steps = (tf / h).ceil() as usize + 1;
    for i in 1..num_steps {
        // Calculate step time
        let t = (i as f64) * h;

        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        forces[(2, elem.nodes.num - 1)] = point_force(t);
        elem.apply_force(&forces);

        // Solve time step
        match solver.step(i, &mut elem, &state) {
            None => {
                print!("{:.3}s: failed to converge \n", t);
                break;
            }
            Some((state_next, energy_incs)) => {
                print!("{:.3}s: converged in {} iterations\n", t, energy_incs.len());
                states.push(state_next.clone());
                file.write_fmt(format_args!("{:?}", t)).expect("fail");
                file.write_fmt(format_args!(",{:?}", energy_incs.len()))
                    .expect("fail");
                for &v in state_next.q.iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");
                state = state_next;
            }
        }
    }
}

#[test]
fn test_cantilever_beam_with_with_sin_load_dirichlet_bc() {
    // Solver parameters
    let rho_inf: f64 = 0.0;
    let tf: f64 = 4.;
    let h: f64 = 0.005;
    let is_dynamic_solve = true;

    let mut elem = build_element(UnitQuaternion::identity());

    //--------------------------------------------------------------------------
    // Create element and solver
    //--------------------------------------------------------------------------

    // Number of state and constraint nodes
    let num_system_nodes = elem.nodes.num;
    let num_constraint_nodes = 0;

    // Create initial state
    let mut state: State = State::new(num_system_nodes, num_constraint_nodes);

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_system_nodes,
        num_constraint_nodes,
        rho_inf,
        h,
        gravity,
        is_dynamic_solve,
    );

    // Create vector to store iteration states
    let mut states: Vec<State> = vec![state.clone()];

    // Sinusoidal force at tip in z direction
    let point_force = |t: f64| -> f64 { 100. * (10. * t).sin() };

    let mut file = std::fs::File::create("q_dbc.csv").expect("file failure");

    // Loop through steps
    let num_steps = (tf / h).ceil() as usize + 1;
    for i in 1..num_steps {
        // Calculate step time
        let t = (i as f64) * h;

        // Apply sinusoidal force at tip in Z direction
        let mut forces: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        forces[(2, elem.nodes.num - 1)] = point_force(t);
        elem.apply_force(&forces);

        // Solve time step
        match solver.step(i, &mut elem, &state) {
            None => {
                print!("{:.3}s: failed to converge \n", t);
                break;
            }
            Some((state_next, iter_data)) => {
                print!("{:.3}s: converged in {} iterations\n", t, iter_data.len());
                states.push(state_next.clone());
                file.write_fmt(format_args!("{:?}", t)).expect("fail");
                file.write_fmt(format_args!(",{:?}", iter_data.len()))
                    .expect("fail");
                for &v in state_next.q.columns(elem.nodes.num - 1, 1).iter() {
                    file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                }
                file.write_all(b"\n").expect("fail");
                state = state_next.clone();
            }
        }
    }
}

#[test]
fn test_rotating_beam() {
    // let mut file = std::fs::File::create("q_rot.csv").expect("file failure");
    let _ = std::fs::remove_dir_all("iter");
    std::fs::create_dir("iter").unwrap();

    // Solver parameters
    let rho_inf: f64 = 0.9;
    let tf: f64 = 10.;
    let h: f64 = 0.01;
    let omega: Vector3 = Vector3::new(0., 0., 0.5); // rad/s
    let is_dynamic_solve = true;

    let omega_init_scale = 0.; // Zero for no velocity initialization

    let rot0: Vector3 = Vector3::new(0., 0., 0.);
    let r0: UnitQuaternion = UnitQuaternion::from_scaled_axis(rot0);

    // Cone angle
    let R_cone = UnitQuaternion::from_scaled_axis(Vector3::new(0., 0., 0.));

    // Create element
    let mut elem = build_element(R_cone);

    // Add hub constraint
    elem.add_rigid_constraint(0, Vector7::from_vec(vec![0., 0., 0., 1., 0., 0., 0.]));

    let num_system_nodes = elem.nodes.num;
    let num_constraint_nodes = elem.constraints.len();

    //--------------------------------------------------------------------------
    // Test solve of element with initial displacement
    //--------------------------------------------------------------------------

    // Gravity loading
    let gravity: Vector3 = Vector3::new(0., 0., 0.);

    // Initial displacement
    let mut Q: Matrix7xX = Matrix7xX::zeros(elem.nodes.num);
    for (i, mut c) in Q.column_iter_mut().enumerate() {
        c.rows_mut(0, 3)
            .copy_from(&(r0 * elem.nodes.x0.column(i) - elem.nodes.x0.column(i)));
        c.rows_mut(3, 4).copy_from(&r0.wijk());
    }

    // Initial velocity
    let mut V: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
    let omega_init = omega * omega_init_scale;
    for (i, mut c) in V.column_iter_mut().enumerate() {
        c.rows_mut(0, 3)
            .copy_from(&(r0 * omega_init.cross(&elem.nodes.x0.column(i))));
        c.rows_mut(3, 3).copy_from(&omega_init);
    }

    // Initial acceleration
    let A: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);

    // Inital state
    let mut state: State =
        State::new_with_initial_state(num_system_nodes, num_constraint_nodes, &Q, &V, &A);

    elem.update_states(&Q, &V, &A, &Vector3::zeros());
    elem.to_vtk().export_ascii("iter/step_000.vtk").unwrap();

    // Create generalized alpha solver
    let mut solver = GeneralizedAlphaSolver::new(
        num_system_nodes,
        num_constraint_nodes,
        rho_inf,
        h,
        gravity,
        is_dynamic_solve,
    );

    let num_steps = (tf / h).ceil() as usize + 1;

    for i in 1..num_steps {
        let t = (i as f64) * h;

        // Prescribe element root displacement
        let rotz = omega * (t) + rot0;
        let r: UnitQuaternion = UnitQuaternion::from_scaled_axis(rotz);
        // elem.q_hub.fixed_rows_mut::<4>(3).copy_from(&r.wijk());
        if elem.constraints.len() > 0 {
            elem.constraints[0]
                .q
                .fixed_rows_mut::<4>(3)
                .copy_from(&r.wijk());
        }

        // let mut forces: Matrix6xX = Matrix6xX::zeros(elem.nodes.num);
        // forces[(2, elem.nodes.num - 1)] = 10.;
        // elem.apply_force(&forces);

        // Solve time step
        match solver.step(i, &mut elem, &state) {
            None => {
                print!("{:.3}s: failed to converge \n", t);
                break;
            }
            Some((state_next, iter_data)) => {
                let rotz_deg = rotz * 180. / PI;
                print!(
                    "Step {:3}, t={:.3}s: converged in {:2} iterations ([{:.3}, {:.3}, {:.3}] rad, [{:.2}, {:.2}, {:.2}] deg)\n",
                    i,
                    t,
                    iter_data.len(),
                    rotz[0], rotz[1], rotz[2],
                    rotz_deg[0], rotz_deg[1], rotz_deg[2]
                );

                // file.write_fmt(format_args!("{:?}", t)).expect("fail");
                // file.write_fmt(format_args!(",{:?}", iter_data.len()))
                //     .expect("fail");
                // for &v in state_next.q.iter() {
                //     file.write_fmt(format_args!(",{:?}", v)).expect("fail");
                // }
                // file.write_all(b"\n").expect("fail");

                // let mut f = std::fs::File::create(format!("iter/step_{:0>3}_x.csv", i)).unwrap();
                // for (i, iter_data) in iter_data.iter().enumerate() {
                //     f.write_fmt(format_args!("{:?}", i + 1)).expect("fail");
                //     for &v in iter_data.x.iter() {
                //         f.write_fmt(format_args!(",{:?}", v)).expect("fail");
                //     }
                //     f.write_all(b"\n").expect("fail");
                // }

                elem.to_vtk()
                    .export_ascii(format!("iter/step_{:0>3}.vtk", i))
                    .unwrap();

                state = state_next.clone();
            }
        }
    }
}
