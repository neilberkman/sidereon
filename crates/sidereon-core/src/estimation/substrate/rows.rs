//! Shared weighted measurement row for the estimation substrate.
//!
//! Every reference strategy ultimately reduces a set of weighted linear rows:
//! SPP feeds `sqrt(w)·(P_meas - P_hat)` residual rows to the trust-region solver,
//! PPP assembles dense `AᵀWA x = AᵀWy` from undifferenced rows, and the RTK
//! filter folds double-difference rows into the information system. The design
//! vector / prefit residual / diagonal weight triple is identical across all
//! three; only the covariance *structure* differs (RTK rows in one block share a
//! reference single difference, see [`super::normal::CovarianceBlock`]).
//!
//! [`ResidualRow`] is that shared triple. The PPP solver row is exactly this
//! type (re-exported as its `Row`); the RTK scratch row additionally carries the
//! single-difference variances the correlated covariance needs, but its design
//! and prefit columns are this same row.

/// One weighted measurement row: design coefficients `h` (length
/// [`super::parameters::ParameterLayout::dim`]), prefit residual `y`, and the
/// diagonal information weight (inverse variance).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualRow {
    pub(crate) h: Vec<f64>,
    pub(crate) y: f64,
    pub(crate) weight: f64,
}

impl ResidualRow {
    /// The `(h, y, weight)` triple in the shape
    /// [`crate::astro::math::linear::normal_equations_weighted`] consumes.
    #[inline]
    pub(crate) fn as_weighted(&self) -> (&[f64], f64, f64) {
        (self.h.as_slice(), self.y, self.weight)
    }
}
