//! GNSS atmospheric correction models.

/// Ionospheric delay models and IONEX grid support.
pub mod ionosphere {
    pub use crate::ionex::{
        galileo_effective_ionisation_level, galileo_nequick_g_native, ionex_slant_delay,
        ionosphere_delay, klobuchar, klobuchar_native, nequick_g_delay_m, nequick_g_stec_tecu,
        regular_tec_grid_delay_xyz, regular_tec_xyz, GalileoNequickCoeffs, GalileoNequickEval,
        Ionex, IonoModel, KlobucharParams, NequickGRayEval, TecGrid, TecGridEpoch,
        TecGridEvalOptions, TecGridShellGeometry,
    };
    pub use crate::rinex_nav::{IonoCorrections, KlobucharAlphaBeta};

    /// Role-oriented alias for a parsed IONEX vertical-TEC grid product.
    pub type IonexGrid = Ionex;
}

/// Tropospheric zenith and slant delay models.
pub mod troposphere {
    pub use crate::tropo::{
        tropo_mapping, tropo_slant, tropo_zenith, tropo_zwd_delay_xyz, zwd_zenith_wet_delay,
        AltitudeClamp, MappingFactors, MappingModel, Met, TropoModel, ZenithDelay, ZwdEpoch,
        ZwdProfile, ZwdSlantOptions,
    };
}

pub use ionosphere::{
    galileo_effective_ionisation_level, galileo_nequick_g_native, ionex_slant_delay,
    ionosphere_delay, klobuchar, klobuchar_native, nequick_g_delay_m, nequick_g_stec_tecu,
    regular_tec_grid_delay_xyz, regular_tec_xyz, GalileoNequickCoeffs, GalileoNequickEval, Ionex,
    IonexGrid, IonoModel, KlobucharParams, NequickGRayEval, TecGrid, TecGridEpoch,
    TecGridEvalOptions, TecGridShellGeometry,
};
pub use troposphere::{
    tropo_mapping, tropo_slant, tropo_zenith, tropo_zwd_delay_xyz, zwd_zenith_wet_delay,
    AltitudeClamp, MappingFactors, MappingModel, Met, TropoModel, ZenithDelay, ZwdEpoch,
    ZwdProfile, ZwdSlantOptions,
};
