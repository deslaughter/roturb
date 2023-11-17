use nalgebra::{Point3, UnitQuaternion};

pub struct Node {
    s: f64,
    position: Point3<f64>,
    rotation: UnitQuaternion<f64>,
}
