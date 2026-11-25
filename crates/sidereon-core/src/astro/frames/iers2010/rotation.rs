use crate::astro::events::eclipse::ad::{ScaledF64, ScaledVec3, Vec3};
use crate::astro::events::eclipse::fast_math::{atan2, sin_cos, sqrt};
use crate::astro::math::mat3::Mat3;

use super::super::orientation::invalid_input;
use super::super::transforms::FrameTransformError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rotation(pub(super) [f64; 4]);

impl Rotation {
    const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0]);

    pub(super) fn celestial([x, y, s]: [f64; 3]) -> Self {
        let radius_squared = x * x + y * y;
        let z_plus_one = 1.0 + sqrt(1.0 - radius_squared);
        let radius = sqrt(radius_squared);
        let (sin, cos) = sin_cos(0.5 * (s + atan2(y, x)));
        let x_plus_radius = x + radius;
        let xc = x_plus_radius * cos;
        let xs = x_plus_radius * sin;
        let yc = y * cos;
        let ys = y * sin;
        let q = [
            z_plus_one * (xc + ys),
            -radius * (yc + xs),
            radius * (xc - ys),
            z_plus_one * (yc - xs),
        ];
        let [a, b, c, d] = q;
        let norm = sqrt(a * a + b * b + c * c + d * d);
        if norm > 0.0 {
            let inverse_norm = 1.0 / norm;
            Self(q.map(|q| inverse_norm * q))
        } else {
            // The CIRF formula loses its common factor on the negative X
            // axis and at the origin. This equivalent quaternion is regular.
            let radius = libm::hypot(x, y);
            let z = sqrt((1.0 - radius) * (1.0 + radius));
            let factor = sqrt(2.0 * (1.0 + z));
            Self([0.5 * factor, -y / factor, x / factor, 0.0]).then(Self::axis(2, -s))
        }
    }

    fn axis(axis: usize, angle: f64) -> Self {
        let (sin, cos) = sin_cos(0.5 * angle);
        let mut vector = [0.0; 3];
        vector[axis] = 1.0;
        Self([cos, sin * vector[0], sin * vector[1], sin * vector[2]])
    }

    // Rotation.compose with FRAME_TRANSFORM convention.
    fn then(self, after: Self) -> Self {
        let [a, b, c, d] = after.0;
        let [w, x, y, z] = self.0;
        Self([
            w * a - (x * b + y * c + z * d),
            x * a + w * b + (y * d - z * c),
            y * a + w * c + (z * b - x * d),
            z * a + w * d + (x * c - y * b),
        ])
    }

    pub(crate) fn components(self) -> [f64; 4] {
        self.0
    }

    pub(crate) fn inverse(self) -> Self {
        let [a, b, c, d] = self.0;
        Self([-a, b, c, d])
    }

    fn apply(self, [x, y, z]: [f64; 3]) -> [f64; 3] {
        let [a, b, c, d] = self.0;
        let s = b * x + c * y + d * z;
        [
            2.0 * (a * (x * a - (c * z - d * y)) + s * b) - x,
            2.0 * (a * (y * a - (d * x - b * z)) + s * c) - y,
            2.0 * (a * (z * a - (b * y - c * x)) + s * d) - z,
        ]
    }

    pub(crate) fn position(self, position: [f64; 3]) -> Result<[f64; 3], FrameTransformError> {
        let result = self.apply(position.map(|v| v + 0.0));
        if result.iter().all(|v| v.is_finite()) {
            Ok(result)
        } else {
            finite_vector(self.apply_scaled(position))
        }
    }

    pub(crate) fn state(
        self,
        position: [f64; 3],
        velocity: [f64; 3],
        rate: [f64; 3],
        inverse: bool,
    ) -> Result<([f64; 3], [f64; 3]), FrameTransformError> {
        let (rotation, rate) = if inverse {
            let inverse = self.inverse();
            (inverse, inverse.apply(rate).map(|v| -v))
        } else {
            (self, rate)
        };
        let p = rotation.apply(position.map(|v| v + 0.0));
        let v = rotation.apply(velocity.map(|v| v + 0.0));
        let [rx, ry, rz] = rate;
        let [x, y, z] = p;
        let transport = Vec3::new(rx, ry, rz).cross(Vec3::new(x, y, z));
        let v = [v[0] - transport.x, v[1] - transport.y, v[2] - transport.z];
        if p.iter().chain(&v).all(|v| v.is_finite()) {
            return Ok((p, v));
        }
        let p = rotation.apply_scaled(position);
        let v = rotation.apply_scaled(velocity);
        let transport = ScaledVec3::from_vec3(Vec3::new(rx, ry, rz)).cross(p);
        Ok((finite_vector(p)?, finite_vector(v.sub(transport))?))
    }

    fn apply_scaled(self, input: [f64; 3]) -> ScaledVec3 {
        let input = input.map(ScaledF64::from_f64);
        let [x, y, z] = self.matrix().map(|row| {
            ScaledF64::lincomb_slice(std::array::from_fn::<_, 3, _>(|i| {
                (ScaledF64::from_f64(row[i]), input[i])
            }))
        });
        ScaledVec3 { x, y, z }
    }

    pub(super) fn matrix(self) -> Mat3 {
        let [a, b, c, d] = self.0;
        [
            [
                2.0 * (a * a + b * b) - 1.0,
                2.0 * (b * c + a * d),
                2.0 * (b * d - a * c),
            ],
            [
                2.0 * (b * c - a * d),
                2.0 * (a * a + c * c) - 1.0,
                2.0 * (c * d + a * b),
            ],
            [
                2.0 * (b * d + a * c),
                2.0 * (c * d - a * b),
                2.0 * (a * a + d * d) - 1.0,
            ],
        ]
    }
}

#[derive(Clone, Copy)]
pub(super) struct Transform {
    pub(super) rotation: Rotation,
    pub(super) rate: [f64; 3],
}

impl Transform {
    const IDENTITY: Self = Self {
        rotation: Rotation::IDENTITY,
        rate: [0.0; 3],
    };

    pub(super) fn new(celestial: Rotation, era: f64, sp: f64, eop: [f64; 6]) -> Self {
        let [xp, yp, _, lod, _, _] = eop;
        let terrestrial = Self {
            rotation: Rotation::axis(0, yp)
                .then(Rotation::axis(1, xp).then(Rotation::axis(2, -sp)))
                .inverse(),
            rate: [0.0; 3],
        };
        let earth = Self {
            rotation: Rotation::axis(2, era),
            rate: [0.0, 0.0, 7.292115146706979e-5 * (1.0 - lod / super::DAY)],
        };
        let celestial = Self {
            rotation: celestial,
            rate: [0.0; 3],
        };
        // Frame.getTransformTo accumulates from each child toward its parent.
        Self::IDENTITY.then(celestial.then(earth.then(terrestrial.then(Self::IDENTITY))))
    }

    fn then(self, after: Self) -> Self {
        let rotated = after.rotation.apply(self.rate);
        Self {
            rotation: self.rotation.then(after.rotation),
            rate: std::array::from_fn(|i| after.rate[i] + rotated[i]),
        }
    }
}

fn finite_vector(vector: ScaledVec3) -> Result<[f64; 3], FrameTransformError> {
    vector
        .to_vec3()
        .map(|v| [v.x, v.y, v.z])
        .ok_or_else(|| invalid_input("state", "transformed components exceed the finite range"))
}
