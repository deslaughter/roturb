use nalgebra;
use nalgebra::{Dyn, U7};

pub use itertools::{izip, Itertools};
pub use std::ops::AddAssign;

pub use std::f64::consts::PI;

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------

/// Matrix (3 x 3)
pub type Matrix3 = nalgebra::Matrix3<f64>;

/// Matrix (6 x 6)
pub type Matrix6 = nalgebra::Matrix6<f64>;

/// Matrix (3 x 4)
pub type Matrix3x4 = nalgebra::Matrix3x4<f64>;

// Matrix of dynamic size
pub type MatrixX = nalgebra::OMatrix<f64, Dyn, Dyn>;

/// Matrix (Nodes x Nodes)
pub type MatrixN = nalgebra::OMatrix<f64, Dyn, Dyn>;

/// Matrix (Nodes x Quaternions)
pub type MatrixNxQ = nalgebra::OMatrix<f64, Dyn, Dyn>;

/// Matrix (DOFs x DOFs)
pub type MatrixD = nalgebra::OMatrix<f64, Dyn, Dyn>;

pub type Matrix1xX = nalgebra::Matrix1xX<f64>;
pub type Matrix3xX = nalgebra::Matrix3xX<f64>;
pub type Matrix4xX = nalgebra::Matrix4xX<f64>;
pub type Matrix6xX = nalgebra::Matrix6xX<f64>;
pub type Matrix7xX = nalgebra::OMatrix<f64, U7, Dyn>;

pub type Vector3 = nalgebra::Vector3<f64>;
pub type Vector4 = nalgebra::Vector4<f64>;
pub type Vector6 = nalgebra::Vector6<f64>;
pub type Vector7 = nalgebra::OVector<f64, U7>;

/// Column vector (Nodes)
pub type VectorN = nalgebra::DVector<f64>;

/// Column vector (Degrees of Freedom)
pub type VectorD = nalgebra::DVector<f64>;

/// Column vector (Quadrature Points)
pub type VectorQ = nalgebra::DVector<f64>;

pub type Quaternion = nalgebra::Quaternion<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

pub type Rotation3 = nalgebra::Rotation3<f64>;

//------------------------------------------------------------------------------
// Traits
//------------------------------------------------------------------------------

pub trait ExtMatrix3 {
    fn axial(&self) -> Vector3;
}

impl ExtMatrix3 for Matrix3 {
    fn axial(&self) -> Vector3 {
        Vector3::new(
            self[(2, 1)] - self[(1, 2)],
            self[(0, 2)] - self[(2, 0)],
            self[(1, 0)] - self[(0, 1)],
        ) / 2.
    }
}

pub trait VecToQuatExt {
    fn as_quaternion(&self) -> Quaternion;
    fn as_unit_quaternion(&self) -> UnitQuaternion;
}

impl VecToQuatExt for Vector4 {
    fn as_quaternion(&self) -> Quaternion {
        Quaternion::new(self[0], self[1], self[2], self[3])
    }
    fn as_unit_quaternion(&self) -> UnitQuaternion {
        let q = self.as_quaternion();
        if q.norm() == 0. {
            UnitQuaternion::identity()
        } else {
            UnitQuaternion::from_quaternion(q)
        }
    }
}

pub trait RotVecExt {
    fn tangent_matrix(&self) -> Matrix3;
    fn tilde(&self) -> Matrix3;
}

impl RotVecExt for Vector3 {
    fn tilde(&self) -> Matrix3 {
        Matrix3::new(
            0.0, -self[2], self[1], self[2], 0.0, -self[0], -self[1], self[0], 0.0,
        )
    }
    fn tangent_matrix(&self) -> Matrix3 {
        let phi = self.magnitude();
        if phi == 0. {
            Matrix3::identity()
        } else {
            Matrix3::identity()
                + (phi.cos() - 1.) / phi.powi(2) * self.tilde()
                + (1. - phi.sin() / phi) * self.tilde() * self.tilde() / phi.powi(2)
        }
        // let psi: Vector3 = self.clone_owned();
        // let phi = psi.norm();
        // let f1 = if phi > 1e-2 {
        //     (phi.cos() - 1.) / phi.powi(2)
        // } else {
        //     -0.5 + phi.powi(2) / 24. - phi.powi(4) / 720.
        // };
        // let f2 = if phi > 1e-4 {
        //     (phi - phi.sin()) / phi.powi(3)
        // } else {
        //     1. / 6. - phi.powi(2) / 120. + phi.powi(4) / 5040.
        // };
        // Matrix3::identity() + f1 * psi.tilde() + f2 * psi.tilde() * psi.tilde()
    }
}

pub trait QuatExt {
    fn E(&self) -> Matrix3x4;
    fn G(&self) -> Matrix3x4;
    fn wijk(&self) -> Vector4;
}

impl QuatExt for UnitQuaternion {
    fn E(&self) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(
            -q1, q0, -q3, q2, // row 1
            -q2, q3, q0, -q1, // row 2
            -q3, -q2, q1, q0, // row 3
        )
    }
    fn G(&self) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(
            -q1, q0, q3, -q2, // row 1
            -q2, -q3, q0, q1, // row 2
            -q3, q2, -q1, q0, // row 3
        )
    }
    fn wijk(&self) -> Vector4 {
        let mut q0 = self.w;
        let mut q = self.vector();
        Vector4::new(q0, q[0], q[1], q[2])
    }
}

impl QuatExt for Quaternion {
    fn E(&self) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(
            -q1, q0, -q3, q2, // row 1
            -q2, q3, q0, -q1, // row 2
            -q3, -q2, q1, q0, // row 3
        )
    }
    fn G(&self) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(
            -q1, q0, q3, -q2, // row 1
            -q2, -q3, q0, q1, // row 2
            -q3, q2, -q1, q0, // row 3
        )
    }
    fn wijk(&self) -> Vector4 {
        let mut q0 = self.w;
        let mut q = self.vector();
        Vector4::new(q0, q[0], q[1], q[2])
    }
}
