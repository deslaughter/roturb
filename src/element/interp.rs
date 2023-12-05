use std::f64::consts::PI;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

//------------------------------------------------------------------------------
// Lagrange Polynomials
//------------------------------------------------------------------------------

pub fn lagrange_polynomial(x: f64, xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(j, &xj)| {
            xs.iter()
                .enumerate()
                .filter(|(m, _)| *m != j)
                .map(|(_, &xm)| (x - xm) / (xj - xm))
                .product()
        })
        .collect()
}

pub fn lagrange_polynomial_derivative(x: f64, xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(j, &sj)| {
            xs.iter()
                .enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(i, &si)| {
                    1.0 / (sj - si)
                        * xs.iter()
                            .enumerate()
                            .filter(|(m, _)| *m != i && *m != j)
                            .map(|(_, &sm)| (x - sm) / (sj - sm))
                            .product::<f64>()
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod test_lagrange {

    use super::*;
    use nalgebra::{dvector, DVector};

    #[test]
    fn test_lagrange_polynomial() {
        let xs = dvector![1.0, 2.0, 3.0];
        let ys = dvector![1.0, 4.0, 9.0];

        let w1 = lagrange_polynomial(1.0, &xs.as_slice());
        let w2 = lagrange_polynomial(2.0, &xs.as_slice());
        let w3 = lagrange_polynomial(3.0, &xs.as_slice());

        assert_eq!(w1, vec![1.0, 0.0, 0.0]);
        assert_eq!(w2, vec![0.0, 1.0, 0.0]);
        assert_eq!(w3, vec![0.0, 0.0, 1.0]);

        assert_eq!(DVector::from_vec(w1).dot(&ys), 1.0);
        assert_eq!(DVector::from_vec(w2).dot(&ys), 4.0);
        assert_eq!(DVector::from_vec(w3).dot(&ys), 9.0);

        let w4 = lagrange_polynomial(1.5, &xs.as_slice());
        assert_eq!(DVector::from_vec(w4).dot(&ys), 1.5 * 1.5);
    }

    #[test]
    fn test_lagrange_polynomial_derivative() {
        let xs = dvector![1.0, 2.0, 3.0];
        let ys = dvector![1.0, 4.0, 9.0];

        let w1 = lagrange_polynomial_derivative(1.0, &xs.as_slice());
        let w2 = lagrange_polynomial_derivative(2.0, &xs.as_slice());
        let w3 = lagrange_polynomial_derivative(3.0, &xs.as_slice());

        assert_eq!(w1, vec![-1.5, 2.0, -0.5]);
        assert_eq!(w2, vec![-0.5, 0.0, 0.5]);
        assert_eq!(w3, vec![0.5, -2.0, 1.5]);

        assert_eq!(DVector::from_vec(w1).dot(&ys), 2.0);
        assert_eq!(DVector::from_vec(w2).dot(&ys), 4.0);
        assert_eq!(DVector::from_vec(w3).dot(&ys), 6.0);

        let w4 = lagrange_polynomial_derivative(1.5, &xs.as_slice());
        assert_eq!(DVector::from_vec(w4).dot(&ys), 2.0 * 1.5);
    }
}

//------------------------------------------------------------------------------
// Legendre Polynomials
//------------------------------------------------------------------------------

pub fn legendre_polynomial(n: usize, xi: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => xi,
        _ => {
            let n_f = n as f64;
            ((2. * n_f - 1.) * xi * legendre_polynomial(n - 1, xi)
                - (n_f - 1.) * legendre_polynomial(n - 2, xi))
                / n_f
        }
    }
}

pub fn legendre_polynomial_derivative_1(n: usize, xi: f64) -> f64 {
    match n {
        0 => 0.,
        1 => 1.,
        2 => 3. * xi,
        _ => {
            (2. * (n as f64) - 1.) * legendre_polynomial(n - 1, xi)
                + legendre_polynomial_derivative_1(n - 2, xi)
        }
    }
}

pub fn legendre_polynomial_derivative_2(n: usize, xi: f64) -> f64 {
    (2. * xi * legendre_polynomial_derivative_1(n, xi)
        - ((n * (n + 1)) as f64) * legendre_polynomial(n, xi))
        / (1. - (xi * xi))
}

pub fn legendre_polynomial_derivative_3(n: usize, xi: f64) -> f64 {
    (4. * xi * legendre_polynomial_derivative_2(n, xi)
        - ((n * (n + 1) - 2) as f64) * legendre_polynomial_derivative_1(n, xi))
        / (1. - (xi * xi))
}

#[cfg(test)]
mod test_legendre {

    use super::*;

    #[test]
    fn test_legendre_polynomial() {
        assert_eq!(legendre_polynomial(0, -1.), 1.);
        assert_eq!(legendre_polynomial(0, 0.), 1.);
        assert_eq!(legendre_polynomial(0, 1.), 1.);

        assert_eq!(legendre_polynomial(1, -1.), -1.);
        assert_eq!(legendre_polynomial(1, 0.), 0.);
        assert_eq!(legendre_polynomial(1, 1.), 1.);

        assert_eq!(legendre_polynomial(2, -1.), 1.);
        assert_eq!(legendre_polynomial(2, 0.), -0.5);
        assert_eq!(legendre_polynomial(2, 1.), 1.);

        assert_eq!(legendre_polynomial(3, -1.), -1.);
        assert_eq!(legendre_polynomial(3, 0.), 0.);
        assert_eq!(legendre_polynomial(3, 1.), 1.);

        assert_eq!(legendre_polynomial(4, -1.), 1.);
        assert_eq!(
            legendre_polynomial(4, -0.6546536707079771),
            -0.4285714285714286
        );
        assert_eq!(legendre_polynomial(4, 0.), 0.375);
        assert_eq!(
            legendre_polynomial(4, 0.6546536707079771),
            -0.4285714285714286
        );
        assert_eq!(legendre_polynomial(4, 1.), 1.);
    }

    #[test]
    fn test_legendre_polynomial_derivative() {
        assert_eq!(legendre_polynomial_derivative_1(0, -1.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(0, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(0, 1.), 0.);

        assert_eq!(legendre_polynomial_derivative_1(1, -1.), 1.);
        assert_eq!(legendre_polynomial_derivative_1(1, 0.), 1.);
        assert_eq!(legendre_polynomial_derivative_1(1, 1.), 1.);

        assert_eq!(legendre_polynomial_derivative_1(2, -1.), -3.);
        assert_eq!(legendre_polynomial_derivative_1(2, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(2, 1.), 3.);

        assert_eq!(legendre_polynomial_derivative_1(3, -1.), 6.);
        assert_eq!(legendre_polynomial_derivative_1(3, 0.), -1.5);
        assert_eq!(legendre_polynomial_derivative_1(3, 1.), 6.);

        assert_eq!(legendre_polynomial_derivative_1(6, -1.), -21.);
        assert_eq!(legendre_polynomial_derivative_1(6, 0.), 0.);
        assert_eq!(legendre_polynomial_derivative_1(6, 1.), 21.);
    }
}

//------------------------------------------------------------------------------
// Gauss Legendre Lobotto Points and Weights
//------------------------------------------------------------------------------

pub fn gauss_legendre_lobotto_points(order: usize) -> Vec<f64> {
    let n = order + 1;
    let n_2 = n / 2;
    let nf = n as f64;
    let mut x = vec![0.0; n];
    // let mut w = vec![0.0; n];
    x[0] = -1.;
    x[n - 1] = 1.;
    for i in 1..n_2 {
        let mut xi = (1. - (3. * (nf - 2.)) / (8. * (nf - 1.).powi(3)))
            * ((4. * (i as f64) + 1.) * PI / (4. * (nf - 1.) + 1.)).cos();
        let mut error = 1.0;
        while error > 1e-16 {
            let y = legendre_polynomial_derivative_1(n - 1, xi);
            let y1 = legendre_polynomial_derivative_2(n - 1, xi);
            let y2 = legendre_polynomial_derivative_3(n - 1, xi);
            let dx = 2. * y * y1 / (2. * y1 * y1 - y * y2);
            xi -= dx;
            error = dx.abs();
        }
        x[i] = -xi;
        x[n - i - 1] = xi;
        // w[i] = 2. / (nf * (nf - 1.) * legendre_polynomial(n - 1, x[i]).powi(2));
        // w[n - i - 1] = w[i];
    }
    if n % 2 != 0 {
        x[n_2] = 0.;
        // w[n_2] = 2.0 / ((nf * (nf - 1.)) * legendre_polynomial(n - 1, x[n_2]).powi(2));
    }
    x
}

#[cfg(test)]
mod test_gll {

    use super::*;

    #[test]
    fn test_gauss_legendre_lobotto_points() {
        assert_eq!(gauss_legendre_lobotto_points(1), vec![-1., 1.]);
        assert_eq!(gauss_legendre_lobotto_points(2), vec![-1., 0., 1.]);
        let p = vec![
            -1.,
            -0.6546536707079771437983,
            0.,
            0.654653670707977143798,
            1.,
        ];
        assert_eq!(gauss_legendre_lobotto_points(4), p);
        let p = vec![
            -1.,
            -0.830223896278567,
            -0.46884879347071423,
            0.,
            0.46884879347071423,
            0.830223896278567,
            1.,
        ];
        assert_eq!(gauss_legendre_lobotto_points(6), p);
    }
}

//------------------------------------------------------------------------------
// Quaternion
//------------------------------------------------------------------------------

pub fn quaternion_from_tangent_twist(tangent: &Vector3<f64>, twist: f64) -> UnitQuaternion<f64> {
    let e1 = tangent.clone();
    let a = if e1[0] > 0. { 1. } else { -1. };
    let e2 = Vector3::new(
        -a * e1[1] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
        a * e1[0] / (e1[0].powi(2) + e1[1].powi(2)).sqrt(),
        0.,
    );
    let e3 = e1.cross(&e2);
    let r0 = Matrix3::from_columns(&[e1, e2, e3]);
    let q_twist = UnitQuaternion::new(e1 * twist * PI / 180.);
    let q0 = UnitQuaternion::from_matrix(&r0);
    q_twist * q0
}

//------------------------------------------------------------------------------
// Integration test
//------------------------------------------------------------------------------

#[cfg(test)]
mod test_integration {

    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector, Matrix3, MatrixXx3, MatrixXx4, UnitQuaternion, Vector4};

    use super::*;

    #[test]
    fn test_shape_functions() {
        // Shape Functions, Derivative of Shape Functions, GLL points for an q-order element
        let p = gauss_legendre_lobotto_points(4);
        assert_eq!(
            p,
            vec![-1., -0.6546536707079772, 0., 0.6546536707079772, 1.]
        );

        // Reference-Line Definition: Here we create a somewhat complex polynomial
        // representation of a line with twist; gives us reference length and curvature to test against
        let s = DVector::from_vec(p).add_scalar(1.) / 2.;

        let fz = |t: f64| -> f64 { t - 2. * t * t };
        let fy = |t: f64| -> f64 { -2. * t + 3. * t * t };
        let fx = |t: f64| -> f64 { 5. * t };
        let ft = |t: f64| -> f64 { 0. * t * t };

        // Node x, y, z, twist along reference line
        let ref_line = MatrixXx4::from_row_iterator(
            s.len(),
            s.iter()
                .flat_map(|&si| vec![fx(si), fy(si), fz(si), ft(si)]),
        );

        // Shape function derivatives at each node along reference line
        let shape_deriv = DMatrix::from_row_iterator(
            s.len(),
            s.len(),
            s.iter()
                .flat_map(|&si| lagrange_polynomial_derivative(si, &s.as_slice())),
        );

        // Tangent vectors at each node along reference line
        let mut ref_tan: MatrixXx3<f64> = shape_deriv
            * MatrixXx3::from_row_iterator(
                s.len(),
                ref_line.row_iter().flat_map(|r| vec![r[0], r[1], r[2]]),
            );
        for mut row in ref_tan.row_iter_mut() {
            row.copy_from(&row.normalize());
        }

        assert_relative_eq!(
            ref_tan,
            MatrixXx3::from_row_iterator(
                5,
                vec![
                    0.9128709291752768,
                    -0.365148371670111,
                    0.1825741858350555,
                    0.9801116185947563,
                    -0.1889578775710052,
                    0.06063114380768645,
                    0.9622504486493763,
                    0.19245008972987512,
                    -0.1924500897298752,
                    0.7994326396775596,
                    0.4738974351647219,
                    -0.3692271327549852,
                    0.7071067811865474,
                    0.5656854249492382,
                    -0.4242640687119285,
                ]
            ),
            epsilon = 1e-15
        );

        // Rotation matrix at each node along reference line
        let ref_q = ref_tan
            .row_iter()
            .zip(ref_line.column(3).iter())
            .map(|(v, &twist)| quaternion_from_tangent_twist(&v.transpose(), twist))
            .collect::<Vec<UnitQuaternion<f64>>>();

        assert_relative_eq!(
            *ref_q[3].to_rotation_matrix().matrix(),
            Matrix3::new(
                0.7994326396775595,
                -0.509929465806723,
                0.3176151673334238,
                0.47389743516472144,
                0.8602162169490122,
                0.18827979456709748,
                -0.3692271327549849,
                0.,
                0.9293391869697156,
            ),
            epsilon = 1e-15
        );

        assert_relative_eq!(
            *ref_q[3].as_vector(),
            Vector4::new(
                -0.049692141629315074, // i
                0.18127630174800594,   // j
                0.25965858850765167,   // k
                0.9472312341234699,    // w
            ),
            epsilon = 1e-15
        );
    }
}
