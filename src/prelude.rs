use nalgebra;
use nalgebra::{Dyn, U7};

pub use itertools::{izip, Itertools};
pub use std::ops::AddAssign;

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------

/// Matrix (3 x 3)
pub type Matrix3 = nalgebra::Matrix3<f64>;

/// Matrix (6 x 6)
pub type Matrix6 = nalgebra::Matrix6<f64>;

/// Matrix (3 x 4)
pub type Matrix3x4 = nalgebra::Matrix3x4<f64>;

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

pub trait TildeExt {
    fn tilde(&self) -> Matrix3;
}

impl TildeExt for Vector3 {
    fn tilde(&self) -> Matrix3 {
        Matrix3::new(
            0.0, -self[2], self[1], self[2], 0.0, -self[0], -self[1], self[0], 0.0,
        )
    }
}

pub trait QuatExt {
    fn F(&self) -> Matrix3x4;
    fn wijk(&self) -> Vector4;
}

impl QuatExt for UnitQuaternion {
    fn F(&self) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (self.w, self.i, self.j, self.k);
        Matrix3x4::new(-q1, q0, -q3, q2, -q2, q3, q0, -q1, -q3, -q2, q1, q0)
    }
    fn wijk(&self) -> Vector4 {
        Vector4::new(self.w, self.i, self.j, self.k)
    }
}