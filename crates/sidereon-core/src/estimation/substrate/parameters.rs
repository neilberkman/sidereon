//! Shared parameter-block layout for the estimation substrate.
//!
//! Each strategy estimates an ordered stack of parameter blocks. The three
//! reference techniques agree on the leading geometry block (a three-component
//! position or baseline) and differ only in which of the clock / zenith-tropo /
//! ambiguity blocks follow and in what count. Today that layout is open-coded as
//! `3 + epochs.len() + ztd + ambiguities` (PPP) or `3 + ambiguities` (RTK) at
//! each solve site. [`ParameterLayout`] gives the block stack one named home so
//! the unknown count and the block offsets are computed in one place.
//!
//! This is descriptive metadata: the layout reproduces the exact dimension each
//! solver already computes, so naming it changes no behavior.

/// The ordered parameter-block stack a strategy estimates. The block order is
/// fixed: the three-component geometry block (position or baseline) first, then
/// the per-epoch receiver clocks, the zenith tropospheric delay, and the
/// ambiguity block. A technique that does not estimate a block leaves its count
/// at zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParameterLayout {
    /// Geometry block: 3 for an absolute position (SPP/PPP) or a baseline (RTK).
    geometry: usize,
    /// Receiver clock unknowns (one per epoch for the static PPP batch; folded
    /// into the geometry solve for SPP/RTK, so zero here).
    clocks: usize,
    /// Zenith tropospheric delay unknowns (PPP wet-delay estimation).
    ztd: usize,
    /// Ambiguity unknowns (RTK single-difference / PPP undifferenced).
    ambiguities: usize,
}

impl ParameterLayout {
    /// SPP: a three-component position plus the per-system receiver clock(s),
    /// folded together into the trust-region solve (`spp::solve`).
    pub(crate) const fn spp(clocks: usize) -> Self {
        Self {
            geometry: 3,
            clocks,
            ztd: 0,
            ambiguities: 0,
        }
    }

    /// RTK sequential/baseline filter: a three-component baseline plus the
    /// single-difference ambiguity block (`rtk_filter` state, `3 + ambiguities`).
    pub(crate) const fn rtk(ambiguities: usize) -> Self {
        Self {
            geometry: 3,
            clocks: 0,
            ztd: 0,
            ambiguities,
        }
    }

    /// Static PPP batch: position, one receiver clock per epoch, the optional
    /// zenith tropospheric delay, and the ambiguity block
    /// (`precise_positioning`, `3 + epochs + ztd + ambiguities`).
    pub(crate) const fn ppp(epoch_clocks: usize, ztd: usize, ambiguities: usize) -> Self {
        Self {
            geometry: 3,
            clocks: epoch_clocks,
            ztd,
            ambiguities,
        }
    }

    /// Total unknown count (the dimension of the normal system).
    pub(crate) const fn dim(&self) -> usize {
        self.geometry + self.clocks + self.ztd + self.ambiguities
    }

    /// Column offset of the ambiguity block (the index of its first column).
    pub(crate) const fn ambiguity_offset(&self) -> usize {
        self.geometry + self.clocks + self.ztd
    }
}

/// Build the dense undifferenced design row `h` for one static-PPP observation
/// against a [`ParameterLayout::ppp`] stack, in the same column order: the three
/// position partials `los_base`, one receiver-clock column per epoch (`1.0` at
/// `epoch_idx`, else `0.0`), the optional zenith-tropo mapping column, then
/// `n_ambiguities` ambiguity columns (`1.0` at `active_ambiguity` and `0.0`
/// elsewhere). `active_ambiguity` is the phase row's own ambiguity index; it is
/// `None` for the code row and for a held-integer fixed solve (which estimates no
/// ambiguity, so passes `n_ambiguities = 0`). This is the one home for the
/// undifferenced design-row layout the float and fixed PPP row builders share.
pub(crate) fn undifferenced_design_row(
    los_base: [f64; 3],
    epoch_idx: usize,
    n_epochs: usize,
    ztd_mapping: Option<f64>,
    n_ambiguities: usize,
    active_ambiguity: Option<usize>,
) -> Vec<f64> {
    let mut row = vec![los_base[0], los_base[1], los_base[2]];
    row.extend((0..n_epochs).map(|idx| if idx == epoch_idx { 1.0 } else { 0.0 }));
    if let Some(mapping) = ztd_mapping {
        row.push(mapping);
    }
    row.extend((0..n_ambiguities).map(|idx| {
        if Some(idx) == active_ambiguity {
            1.0
        } else {
            0.0
        }
    }));
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_match_open_coded_unknown_counts() {
        // SPP: position + clocks.
        assert_eq!(ParameterLayout::spp(1).dim(), 4);
        assert_eq!(ParameterLayout::spp(2).dim(), 5);
        // RTK: baseline + ambiguities (FilterState `3 + ambiguities`).
        assert_eq!(ParameterLayout::rtk(0).dim(), 3);
        assert_eq!(ParameterLayout::rtk(7).dim(), 10);
        // PPP: position + epoch clocks + ztd + ambiguities
        // (`3 + epochs + ztd + ambiguities`): 3 + 5 + 0 + 4 and 3 + 5 + 1 + 4.
        assert_eq!(ParameterLayout::ppp(5, 0, 4).dim(), 12);
        assert_eq!(ParameterLayout::ppp(5, 1, 4).dim(), 13);
    }

    #[test]
    fn ambiguity_offset_follows_geometry_clocks_ztd() {
        // PPP ambiguities start after position + epoch clocks + ztd, matching
        // `start = 3 + epochs + ztd` in the fixed solver.
        assert_eq!(ParameterLayout::ppp(5, 1, 4).ambiguity_offset(), 3 + 5 + 1);
        // RTK ambiguities start right after the baseline (column 3).
        assert_eq!(ParameterLayout::rtk(7).ambiguity_offset(), 3);
    }

    #[test]
    fn undifferenced_design_row_lays_columns_in_layout_order() {
        // Float phase row, 2 epochs, ztd on, 3 ambiguities, this obs is amb 2.
        let float_phase = undifferenced_design_row([-0.1, -0.2, -0.3], 1, 2, Some(1.5), 3, Some(2));
        assert_eq!(
            float_phase,
            vec![-0.1, -0.2, -0.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0]
        );
        // The matching code row clears the ambiguity column.
        let float_code = undifferenced_design_row([-0.1, -0.2, -0.3], 1, 2, Some(1.5), 3, None);
        assert_eq!(
            float_code,
            vec![-0.1, -0.2, -0.3, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0]
        );
        // Fixed row: ambiguities held, tropo off, so neither block appears.
        let fixed = undifferenced_design_row([-0.1, -0.2, -0.3], 0, 2, None, 0, None);
        assert_eq!(fixed, vec![-0.1, -0.2, -0.3, 1.0, 0.0]);
    }
}
