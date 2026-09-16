//! GNSS atmospheric correction models.

/// Ionospheric delay models and IONEX grid support.
pub mod ionosphere {
    pub use crate::ionex::{
        galileo_effective_ionisation_level, galileo_nequick_g_native, ionex_slant_delay,
        ionex_slant_delay_results, ionex_slant_delay_with_policy, ionex_slant_delays,
        ionosphere_delay, klobuchar, klobuchar_native, nequick_g_delay_m, nequick_g_stec_tecu,
        regular_tec_grid_delay_xyz, regular_tec_grid_delay_xyz_with_policy, regular_tec_xyz,
        regular_tec_xyz_with_policy, GalileoNequickCoeffs, GalileoNequickEval, Ionex,
        IonexAssumedMapping, IonexCoverageError, IonexCoveragePolicy, IonexHeader,
        IonexMappingDeclaration, IonexMappingFunction, IonexMappingPolicy, IonexMissingNodePolicy,
        IonexMissingNodes, IonexNodeGap, IonexSlantDelayEvaluation, IonexSlantDelayStatus,
        IonexSlantPolicy, IonexSlantRefusal, IonexSlantRequest, IonexWarning, IonoModel,
        KlobucharParams, NequickGRayEval, TecGrid, TecGridEpoch, TecGridError, TecGridEvalOptions,
        TecGridEvaluation, TecGridSamples, TecGridShellGeometry, TecSample, TecSamplesError,
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
        ZwdProfile, ZwdSlantOptions, NIELL_MIN_MAPPING_ELEVATION_RAD,
        TROPO_MIN_MAPPING_ELEVATION_RAD,
    };
}

pub use ionosphere::{
    galileo_effective_ionisation_level, galileo_nequick_g_native, ionex_slant_delay,
    ionex_slant_delay_results, ionex_slant_delay_with_policy, ionex_slant_delays, ionosphere_delay,
    klobuchar, klobuchar_native, nequick_g_delay_m, nequick_g_stec_tecu,
    regular_tec_grid_delay_xyz, regular_tec_grid_delay_xyz_with_policy, regular_tec_xyz,
    regular_tec_xyz_with_policy, GalileoNequickCoeffs, GalileoNequickEval, Ionex,
    IonexAssumedMapping, IonexCoverageError, IonexCoveragePolicy, IonexGrid, IonexHeader,
    IonexMappingDeclaration, IonexMappingFunction, IonexMappingPolicy, IonexMissingNodePolicy,
    IonexMissingNodes, IonexNodeGap, IonexSlantDelayEvaluation, IonexSlantDelayStatus,
    IonexSlantPolicy, IonexSlantRefusal, IonexSlantRequest, IonexWarning, IonoModel,
    KlobucharParams, NequickGRayEval, TecGrid, TecGridEpoch, TecGridError, TecGridEvalOptions,
    TecGridEvaluation, TecGridSamples, TecGridShellGeometry, TecSample, TecSamplesError,
};
pub use troposphere::{
    tropo_mapping, tropo_slant, tropo_zenith, tropo_zwd_delay_xyz, zwd_zenith_wet_delay,
    AltitudeClamp, MappingFactors, MappingModel, Met, TropoModel, ZenithDelay, ZwdEpoch,
    ZwdProfile, ZwdSlantOptions, NIELL_MIN_MAPPING_ELEVATION_RAD, TROPO_MIN_MAPPING_ELEVATION_RAD,
};
