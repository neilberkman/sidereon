//! Ephemeris products and satellite orbit/clock evaluation.
//!
//! This is the main public home for loaded GNSS ephemeris data. Use
//! [`Sp3`] for precise SP3 products and [`BroadcastEphemeris`] for broadcast
//! navigation products parsed from RINEX NAV files (or built from records decoded
//! off the air with [`BroadcastRecord::from_lnav`]). Both implement
//! [`EphemerisSource`] so they can feed [`crate::positioning::solve`]; the
//! broadcast-only real-time/offline path is
//! [`solve_broadcast`](crate::positioning::solve_broadcast) and the
//! precise-with-broadcast-fallback path is
//! [`solve_with_fallback`](crate::positioning::solve_with_fallback).

pub use crate::broadcast::{
    eccentric_anomaly, relativistic_clock_correction_s, satellite_clock_offset_s,
    satellite_position_ecef, satellite_state, ClockOffset, ClockPolynomial, ConstellationConstants,
    EccentricAnomaly, KeplerianElements, OrbitState, SatelliteState,
};
pub use crate::rinex_nav::{
    is_beidou_geo, BroadcastGroupDelayTerm, BroadcastGroupDelays, BroadcastRecord, GlonassRecord,
    IonoCorrections, KlobucharAlphaBeta, LnavRecordError, NavMessage,
};
pub use crate::sp3::{
    align_clock_reference, clock_reference_offset, merge, AgreementMetric, ClockReferenceOffset,
    EpochAgreement, MergeCombine, MergeFlag, MergeOptions, MergeReport, PreciseEphemerisSample,
    PreciseEphemerisSamples, PreciseSamplesError, Sp3, Sp3DataType, Sp3Flags, Sp3Header, Sp3State,
    Sp3TimeSystem, Sp3Version,
};
pub use crate::spp::EphemerisSource;
use crate::GnssSystem;

/// Broadcast navigation ephemeris store selected by satellite and query epoch.
///
/// The underlying implementation type is `BroadcastStore`; the public alias
/// names the role rather than the storage detail.
pub type BroadcastEphemeris = crate::rinex_nav::BroadcastStore;

/// Acronym-preserving alias for users who prefer the format name spelling.
///
/// Rust item names normally use `Sp3`; this alias keeps `SP3` available without
/// making the implementation type fight Rust naming conventions.
#[allow(clippy::upper_case_acronyms)]
pub type SP3 = Sp3;

/// Select a broadcast TGD/BGD value from a parsed delay set.
///
/// This is a function-form accessor for [`BroadcastGroupDelays::get`], useful
/// when composing the group-delay term independently from a full broadcast clock
/// evaluation.
pub const fn broadcast_group_delay_s(
    delays: &BroadcastGroupDelays,
    term: BroadcastGroupDelayTerm,
) -> Option<f64> {
    delays.get(term)
}

/// Select the broadcast group delay used by a navigation message's clock model.
///
/// This is a function-form accessor for [`BroadcastGroupDelays::for_message`].
/// It returns `None` when the supplied delay set does not carry a term for the
/// requested constellation/message pair.
pub const fn broadcast_message_group_delay_s(
    delays: BroadcastGroupDelays,
    system: GnssSystem,
    message: NavMessage,
) -> Option<f64> {
    delays.for_message(system, message)
}

/// Group delay selected by a parsed broadcast navigation record's clock model.
///
/// This is a function-form accessor for
/// [`BroadcastRecord::broadcast_clock_group_delay_s`]. It returns zero when the
/// record carries no applicable TGD/BGD term, matching the broadcast clock
/// evaluator.
pub fn broadcast_record_group_delay_s(record: &BroadcastRecord) -> f64 {
    record.broadcast_clock_group_delay_s()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_group_delay_free_functions_delegate_to_delay_set() {
        let gps = BroadcastGroupDelays::gps_lnav(-2.3e-9);
        assert_eq!(
            broadcast_group_delay_s(&gps, BroadcastGroupDelayTerm::GpsTgd),
            Some(-2.3e-9)
        );
        assert_eq!(
            broadcast_message_group_delay_s(gps, GnssSystem::Gps, NavMessage::GpsLnav),
            Some(-2.3e-9)
        );

        let galileo = BroadcastGroupDelays::galileo(1.0e-9, 2.0e-9);
        assert_eq!(
            broadcast_message_group_delay_s(galileo, GnssSystem::Galileo, NavMessage::GalileoFnav),
            Some(1.0e-9)
        );
        assert_eq!(
            broadcast_message_group_delay_s(galileo, GnssSystem::Galileo, NavMessage::GalileoInav),
            Some(2.0e-9)
        );
    }

    #[test]
    fn broadcast_record_group_delay_free_function_delegates_to_record() {
        let record = BroadcastRecord {
            satellite_id: crate::GnssSatelliteId::new(GnssSystem::Galileo, 1)
                .expect("valid satellite"),
            message: NavMessage::GalileoInav,
            week: 2_400,
            toe: crate::astro::time::model::GnssWeekTow::new(
                crate::astro::time::model::TimeScale::Gst,
                2_400,
                100_000.0,
            )
            .expect("valid toe"),
            toc: crate::astro::time::model::GnssWeekTow::new(
                crate::astro::time::model::TimeScale::Gst,
                2_400,
                100_000.0,
            )
            .expect("valid toc"),
            elements: KeplerianElements {
                sqrt_a: 5_440.0,
                e: 0.01,
                m0: 0.1,
                delta_n: 0.0,
                omega0: 0.2,
                i0: 0.94,
                omega: 0.3,
                omega_dot: -8.0e-9,
                idot: 0.0,
                cuc: 0.0,
                cus: 0.0,
                crc: 0.0,
                crs: 0.0,
                cic: 0.0,
                cis: 0.0,
                toe_sow: 100_000.0,
            },
            clock: ClockPolynomial {
                af0: 0.0,
                af1: 0.0,
                af2: 0.0,
                toc_sow: 100_000.0,
            },
            group_delays: BroadcastGroupDelays::galileo(1.0e-9, 2.5e-9),
            sv_health: 0.0,
            sv_accuracy_m: 1.0,
            fit_interval_s: None,
        };

        assert_eq!(
            broadcast_record_group_delay_s(&record).to_bits(),
            record.broadcast_clock_group_delay_s().to_bits()
        );
        assert_eq!(
            broadcast_record_group_delay_s(&record).to_bits(),
            2.5e-9_f64.to_bits()
        );
    }
}
