//! Data product filename, cache path, and archive URL catalog.
//!
//! This module is sans-IO: it performs no network access, reads no files, and
//! writes no cache entries. It only turns cataloged product inputs into
//! canonical archive filenames, URLs, cache relative paths, and deterministic
//! converted bytes for pure terrain ingestion.

use core::fmt;
use core::str::FromStr;

use crate::astro::time::civil::{civil_from_julian_day_number, day_of_year_int, days_in_month};
use crate::astro::time::gnss::{week_epoch_julian_day_number, week_from_calendar};
use crate::astro::time::model::TimeScale;
use crate::astro::time::scales::julian_day_number;
use crate::terrain;

/// Analysis-center code supported by the data-product catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnalysisCenter {
    /// `igs`.
    Igs,
    /// `cod_rap`.
    CodRap,
    /// `cod_prd1`.
    CodPrd1,
    /// `cod_prd2`.
    CodPrd2,
    /// `esa`.
    Esa,
    /// `cod`.
    Cod,
    /// `gfz`.
    Gfz,
    /// `igs_ult`.
    IgsUlt,
    /// `cod_ult`.
    CodUlt,
    /// `esa_ult`.
    EsaUlt,
    /// `gfz_ult`.
    GfzUlt,
}

impl AnalysisCenter {
    /// The lower-case catalog code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Igs => "igs",
            Self::CodRap => "cod_rap",
            Self::CodPrd1 => "cod_prd1",
            Self::CodPrd2 => "cod_prd2",
            Self::Esa => "esa",
            Self::Cod => "cod",
            Self::Gfz => "gfz",
            Self::IgsUlt => "igs_ult",
            Self::CodUlt => "cod_ult",
            Self::EsaUlt => "esa_ult",
            Self::GfzUlt => "gfz_ult",
        }
    }

    /// Parse a lower-case catalog code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "igs" => Some(Self::Igs),
            "cod_rap" => Some(Self::CodRap),
            "cod_prd1" => Some(Self::CodPrd1),
            "cod_prd2" => Some(Self::CodPrd2),
            "esa" => Some(Self::Esa),
            "cod" => Some(Self::Cod),
            "gfz" => Some(Self::Gfz),
            "igs_ult" => Some(Self::IgsUlt),
            "cod_ult" => Some(Self::CodUlt),
            "esa_ult" => Some(Self::EsaUlt),
            "gfz_ult" => Some(Self::GfzUlt),
            _ => None,
        }
    }
}

impl fmt::Display for AnalysisCenter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl FromStr for AnalysisCenter {
    type Err = DataCatalogError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_code(s).ok_or_else(|| DataCatalogError::UnknownCenter(s.to_string()))
    }
}

/// Product type supported by the data-product catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProductType {
    /// Precise orbit SP3.
    Sp3,
    /// RINEX clock.
    Clk,
    /// Merged broadcast navigation.
    Nav,
    /// IONEX global ionosphere map.
    Ionex,
}

impl ProductType {
    /// The lower-case product code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Sp3 => "sp3",
            Self::Clk => "clk",
            Self::Nav => "nav",
            Self::Ionex => "ionex",
        }
    }

    /// Parse a lower-case product code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "sp3" => Some(Self::Sp3),
            "clk" => Some(Self::Clk),
            "nav" => Some(Self::Nav),
            "ionex" => Some(Self::Ionex),
            _ => None,
        }
    }
}

impl fmt::Display for ProductType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl FromStr for ProductType {
    type Err = DataCatalogError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_code(s).ok_or_else(|| DataCatalogError::UnknownProductType(s.to_string()))
    }
}

/// CelesTrak space-weather product served by the data catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpaceWeatherProduct {
    /// `SW-All.csv`: full history plus daily and monthly predictions.
    All,
    /// `SW-Last5Years.csv`: observed rolling window.
    Last5Years,
}

impl SpaceWeatherProduct {
    /// The lower-case catalog code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::All => "sw_all",
            Self::Last5Years => "sw_last5",
        }
    }

    /// Parse a lower-case catalog code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "sw_all" => Some(Self::All),
            "sw_last5" => Some(Self::Last5Years),
            _ => None,
        }
    }
}

impl fmt::Display for SpaceWeatherProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl FromStr for SpaceWeatherProduct {
    type Err = DataCatalogError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_code(s).ok_or_else(|| DataCatalogError::UnknownProductType(s.to_string()))
    }
}

/// Archive transport protocol recorded by the catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveProtocol {
    /// HTTP.
    Http,
    /// HTTPS.
    Https,
}

impl ArchiveProtocol {
    /// URI scheme text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
        }
    }
}

/// Archive compression for a cataloged product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveCompression {
    /// Archive URL has a `.gz` suffix.
    Gzip,
    /// Archive URL is the plain product filename.
    None,
}

impl ArchiveCompression {
    /// Catalog text for the compression format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::None => "none",
        }
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::Gzip => ".gz",
            Self::None => "",
        }
    }
}

/// Directory layout used below an archive root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveLayout {
    /// `rapid/w<gps-week>`.
    GfzRapidWeek,
    /// `ultra/w<gps-week>`.
    GfzUltraWeek,
    /// `<gps-week>`.
    GpsWeek,
    /// `products/<gps-week>`.
    BkgProductsWeek,
    /// `BRDC/<year>/<day-of-year>`.
    BkgBrdcYearDoy,
    /// `obs/<year>/<day-of-year>`.
    BkgObsYearDoy,
    /// `CODE_MGEX/CODE/<year>`.
    AiubCodeMgexYear,
    /// `CODE/<year>`.
    AiubCodeYear,
    /// `CODE`.
    AiubCodeRoot,
}

/// Product filename convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductFilenameKind {
    /// `TOKEN_DATE_LEN_SAMPLE_CODE.EXT`.
    Sampled,
    /// `TOKEN_R_DATE_LEN_CODE.ext`.
    Nav,
}

/// Product-type filename convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductTypeConvention {
    /// Product type.
    pub product_type: ProductType,
    /// Filename content code, for example `ORB`.
    pub content_code: &'static str,
    /// Filename extension, preserving archive case.
    pub extension: &'static str,
    /// Filename convention.
    pub kind: ProductFilenameKind,
}

/// Per-center convention for one product type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CenterProductConvention {
    /// Product type.
    pub product_type: ProductType,
    /// IGS long-name token prefix.
    pub token: &'static str,
    /// Directory layout under the archive root.
    pub layout: ArchiveLayout,
    /// Product span token.
    pub span: &'static str,
    /// Default sampling token.
    pub default_sample: &'static str,
    /// Archive compression.
    pub compression: ArchiveCompression,
}

/// Static catalog entry for one analysis-center code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CenterCatalogEntry {
    /// Analysis-center code.
    pub center: AnalysisCenter,
    /// Lower-case catalog code.
    pub code: &'static str,
    /// Archive URI scheme.
    pub protocol: ArchiveProtocol,
    /// Archive host.
    pub host: &'static str,
    /// Archive root URL without trailing slash.
    pub root_url: &'static str,
    /// Product conventions served by this center.
    pub products: &'static [CenterProductConvention],
    /// Valid issue times for sub-daily products.
    pub issues: &'static [&'static str],
}

/// Static catalog entry for one terrain source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainSourceEntry {
    /// Archive URI scheme.
    pub protocol: ArchiveProtocol,
    /// Archive host.
    pub host: &'static str,
    /// Archive compression.
    pub compression: ArchiveCompression,
    /// Archive root URL without trailing slash.
    pub root_url: &'static str,
}

/// Static catalog entry for the CelesTrak space-weather source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceWeatherSourceEntry {
    /// Archive URI scheme.
    pub protocol: ArchiveProtocol,
    /// Archive host.
    pub host: &'static str,
    /// Archive compression.
    pub compression: ArchiveCompression,
    /// Archive root URL without trailing slash.
    pub root_url: &'static str,
}

/// Product pair that is intentionally not offered because no open mirror exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoOpenMirrorProduct {
    /// Analysis-center code.
    pub center: &'static str,
    /// Product type code.
    pub product_type: &'static str,
}

const PRODUCT_TYPE_CONVENTIONS: [ProductTypeConvention; 4] = [
    ProductTypeConvention {
        product_type: ProductType::Sp3,
        content_code: "ORB",
        extension: "SP3",
        kind: ProductFilenameKind::Sampled,
    },
    ProductTypeConvention {
        product_type: ProductType::Clk,
        content_code: "CLK",
        extension: "CLK",
        kind: ProductFilenameKind::Sampled,
    },
    ProductTypeConvention {
        product_type: ProductType::Nav,
        content_code: "MN",
        extension: "rnx",
        kind: ProductFilenameKind::Nav,
    },
    ProductTypeConvention {
        product_type: ProductType::Ionex,
        content_code: "GIM",
        extension: "INX",
        kind: ProductFilenameKind::Sampled,
    },
];

const COD_RAP_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Ionex,
    token: "COD0OPSRAP",
    layout: ArchiveLayout::AiubCodeRoot,
    span: "01D",
    default_sample: "01H",
    compression: ArchiveCompression::Gzip,
}];

const COD_PRD_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Ionex,
    token: "COD0OPSPRD",
    layout: ArchiveLayout::AiubCodeRoot,
    span: "01D",
    default_sample: "01H",
    compression: ArchiveCompression::Gzip,
}];

const ESA_PRODUCTS: [CenterProductConvention; 3] = [
    CenterProductConvention {
        product_type: ProductType::Sp3,
        token: "ESA0MGNFIN",
        layout: ArchiveLayout::GpsWeek,
        span: "01D",
        default_sample: "05M",
        compression: ArchiveCompression::Gzip,
    },
    CenterProductConvention {
        product_type: ProductType::Clk,
        token: "ESA0MGNFIN",
        layout: ArchiveLayout::GpsWeek,
        span: "01D",
        default_sample: "30S",
        compression: ArchiveCompression::Gzip,
    },
    CenterProductConvention {
        product_type: ProductType::Ionex,
        token: "ESA0OPSFIN",
        layout: ArchiveLayout::GpsWeek,
        span: "01D",
        default_sample: "02H",
        compression: ArchiveCompression::Gzip,
    },
];

const COD_PRODUCTS: [CenterProductConvention; 3] = [
    CenterProductConvention {
        product_type: ProductType::Sp3,
        token: "COD0MGXFIN",
        layout: ArchiveLayout::AiubCodeMgexYear,
        span: "01D",
        default_sample: "05M",
        compression: ArchiveCompression::Gzip,
    },
    CenterProductConvention {
        product_type: ProductType::Clk,
        token: "COD0MGXFIN",
        layout: ArchiveLayout::AiubCodeMgexYear,
        span: "01D",
        default_sample: "30S",
        compression: ArchiveCompression::Gzip,
    },
    CenterProductConvention {
        product_type: ProductType::Ionex,
        token: "COD0OPSFIN",
        layout: ArchiveLayout::AiubCodeYear,
        span: "01D",
        default_sample: "01H",
        compression: ArchiveCompression::Gzip,
    },
];

const GFZ_PRODUCTS: [CenterProductConvention; 2] = [
    CenterProductConvention {
        product_type: ProductType::Sp3,
        token: "GFZ0OPSRAP",
        layout: ArchiveLayout::GfzRapidWeek,
        span: "01D",
        default_sample: "15M",
        compression: ArchiveCompression::Gzip,
    },
    CenterProductConvention {
        product_type: ProductType::Clk,
        token: "GFZ0OPSRAP",
        layout: ArchiveLayout::GfzRapidWeek,
        span: "01D",
        default_sample: "30S",
        compression: ArchiveCompression::Gzip,
    },
];

const IGS_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Nav,
    token: "BRDC00WRD",
    layout: ArchiveLayout::BkgBrdcYearDoy,
    span: "01D",
    default_sample: "01D",
    compression: ArchiveCompression::Gzip,
}];

const IGS_ULT_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Sp3,
    token: "IGS0OPSULT",
    layout: ArchiveLayout::BkgProductsWeek,
    span: "02D",
    default_sample: "15M",
    compression: ArchiveCompression::Gzip,
}];

const COD_ULT_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Sp3,
    token: "COD0OPSULT",
    layout: ArchiveLayout::AiubCodeRoot,
    span: "01D",
    default_sample: "05M",
    compression: ArchiveCompression::None,
}];

const ESA_ULT_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Sp3,
    token: "ESA0OPSULT",
    layout: ArchiveLayout::GpsWeek,
    span: "02D",
    default_sample: "05M",
    compression: ArchiveCompression::Gzip,
}];

const GFZ_ULT_PRODUCTS: [CenterProductConvention; 1] = [CenterProductConvention {
    product_type: ProductType::Sp3,
    token: "GFZ0OPSULT",
    layout: ArchiveLayout::GfzUltraWeek,
    span: "02D",
    default_sample: "05M",
    compression: ArchiveCompression::Gzip,
}];

const OPSULT_ISSUES: [&str; 4] = ["0000", "0600", "1200", "1800"];
const COD_ULT_ISSUES: [&str; 1] = ["0000"];
const GFZ_ULT_ISSUES: [&str; 8] = [
    "0000", "0300", "0600", "0900", "1200", "1500", "1800", "2100",
];

const CENTER_ORDER: [AnalysisCenter; 11] = [
    AnalysisCenter::CodRap,
    AnalysisCenter::CodPrd1,
    AnalysisCenter::CodPrd2,
    AnalysisCenter::Igs,
    AnalysisCenter::Esa,
    AnalysisCenter::Cod,
    AnalysisCenter::Gfz,
    AnalysisCenter::IgsUlt,
    AnalysisCenter::CodUlt,
    AnalysisCenter::EsaUlt,
    AnalysisCenter::GfzUlt,
];

const CATALOG: [CenterCatalogEntry; 11] = [
    CenterCatalogEntry {
        center: AnalysisCenter::CodRap,
        code: "cod_rap",
        protocol: ArchiveProtocol::Http,
        host: "ftp.aiub.unibe.ch",
        root_url: "http://ftp.aiub.unibe.ch",
        products: &COD_RAP_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::CodPrd1,
        code: "cod_prd1",
        protocol: ArchiveProtocol::Http,
        host: "ftp.aiub.unibe.ch",
        root_url: "http://ftp.aiub.unibe.ch",
        products: &COD_PRD_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::CodPrd2,
        code: "cod_prd2",
        protocol: ArchiveProtocol::Http,
        host: "ftp.aiub.unibe.ch",
        root_url: "http://ftp.aiub.unibe.ch",
        products: &COD_PRD_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::Igs,
        code: "igs",
        protocol: ArchiveProtocol::Https,
        host: "igs.bkg.bund.de",
        root_url: "https://igs.bkg.bund.de/root_ftp/IGS",
        products: &IGS_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::Esa,
        code: "esa",
        protocol: ArchiveProtocol::Https,
        host: "navigation-office.esa.int",
        root_url: "https://navigation-office.esa.int/products/gnss-products",
        products: &ESA_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::Cod,
        code: "cod",
        protocol: ArchiveProtocol::Http,
        host: "ftp.aiub.unibe.ch",
        root_url: "http://ftp.aiub.unibe.ch",
        products: &COD_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::Gfz,
        code: "gfz",
        protocol: ArchiveProtocol::Https,
        host: "isdc-data.gfz.de",
        root_url: "https://isdc-data.gfz.de/gnss/products",
        products: &GFZ_PRODUCTS,
        issues: &[],
    },
    CenterCatalogEntry {
        center: AnalysisCenter::IgsUlt,
        code: "igs_ult",
        protocol: ArchiveProtocol::Https,
        host: "igs.bkg.bund.de",
        root_url: "https://igs.bkg.bund.de/root_ftp/IGS",
        products: &IGS_ULT_PRODUCTS,
        issues: &OPSULT_ISSUES,
    },
    CenterCatalogEntry {
        center: AnalysisCenter::CodUlt,
        code: "cod_ult",
        protocol: ArchiveProtocol::Https,
        host: "www.aiub.unibe.ch",
        // AIUB retired the old ftp.aiub.unibe.ch HTTP tree. Its public file
        // browser links products through this stable HTTPS download surface,
        // which redirects to the current object store.
        root_url: "https://www.aiub.unibe.ch/download",
        products: &COD_ULT_PRODUCTS,
        issues: &COD_ULT_ISSUES,
    },
    CenterCatalogEntry {
        center: AnalysisCenter::EsaUlt,
        code: "esa_ult",
        protocol: ArchiveProtocol::Https,
        host: "navigation-office.esa.int",
        root_url: "https://navigation-office.esa.int/products/gnss-products",
        products: &ESA_ULT_PRODUCTS,
        issues: &OPSULT_ISSUES,
    },
    CenterCatalogEntry {
        center: AnalysisCenter::GfzUlt,
        code: "gfz_ult",
        protocol: ArchiveProtocol::Https,
        host: "isdc-data.gfz.de",
        root_url: "https://isdc-data.gfz.de/gnss/products",
        products: &GFZ_ULT_PRODUCTS,
        issues: &GFZ_ULT_ISSUES,
    },
];

const SKADI_SOURCE: TerrainSourceEntry = TerrainSourceEntry {
    protocol: ArchiveProtocol::Https,
    host: "s3.amazonaws.com",
    compression: ArchiveCompression::Gzip,
    root_url: "https://s3.amazonaws.com/elevation-tiles-prod",
};

const CELESTRAK_SPACE_WEATHER_SOURCE: SpaceWeatherSourceEntry = SpaceWeatherSourceEntry {
    protocol: ArchiveProtocol::Https,
    host: "celestrak.org",
    compression: ArchiveCompression::None,
    root_url: "https://celestrak.org/SpaceData",
};

const ALLOWED_HOSTS: [&str; 7] = [
    "ftp.aiub.unibe.ch",
    "www.aiub.unibe.ch",
    "navigation-office.esa.int",
    "isdc-data.gfz.de",
    "igs.bkg.bund.de",
    "s3.amazonaws.com",
    "celestrak.org",
];

const NO_OPEN_MIRRORS: [NoOpenMirrorProduct; 7] = [
    NoOpenMirrorProduct {
        center: "grg",
        product_type: "sp3",
    },
    NoOpenMirrorProduct {
        center: "grg",
        product_type: "clk",
    },
    NoOpenMirrorProduct {
        center: "wum",
        product_type: "sp3",
    },
    NoOpenMirrorProduct {
        center: "wum",
        product_type: "clk",
    },
    NoOpenMirrorProduct {
        center: "grg_ult",
        product_type: "sp3",
    },
    NoOpenMirrorProduct {
        center: "grg_ult",
        product_type: "clk",
    },
    NoOpenMirrorProduct {
        center: "igs",
        product_type: "ionex",
    },
];

/// Error returned by the pure data-product catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataCatalogError {
    /// Unknown analysis-center code.
    UnknownCenter(String),
    /// Unknown product type code.
    UnknownProductType(String),
    /// The center does not serve the requested product type.
    UnsupportedProduct {
        /// Analysis center.
        center: AnalysisCenter,
        /// Product type.
        product_type: ProductType,
    },
    /// The product has no verified anonymous HTTP(S) mirror.
    NoOpenMirror {
        /// Analysis-center code.
        center: String,
        /// Product type code.
        product_type: String,
    },
    /// Bad civil date.
    InvalidDate {
        /// Year.
        year: i32,
        /// Month.
        month: u8,
        /// Day.
        day: u8,
    },
    /// Date cannot be represented by this API.
    DateOutOfRange,
    /// Date precedes the GPS week epoch.
    DateBeforeGpsEpoch(ProductDate),
    /// GPS day-of-week must be `0..=6`.
    InvalidGpsDayOfWeek(u8),
    /// Sampling token is not `NNX` with an upper-case unit.
    InvalidSample(String),
    /// Issue time is malformed.
    InvalidIssue(String),
    /// The center requires an issue time.
    MissingIssue {
        /// Analysis center.
        center: AnalysisCenter,
    },
    /// The center does not use issue times.
    UnexpectedIssue {
        /// Analysis center.
        center: AnalysisCenter,
    },
    /// Issue time is valid text but not published by this center.
    UnsupportedIssue {
        /// Analysis center.
        center: AnalysisCenter,
        /// Issue time.
        issue: String,
    },
    /// The target datetime was invalid.
    InvalidDateTime {
        /// Hour.
        hour: u8,
        /// Minute.
        minute: u8,
        /// Second.
        second: u8,
    },
    /// No ultra-rapid issue exists at or before the requested target.
    NoUltraIssue,
    /// No available ultra-rapid issue exists at or before the requested target.
    NoAvailableUltraIssue,
    /// Station identifier is not a 9-character upper-case alphanumeric token.
    InvalidStation(String),
    /// Terrain lookup coordinate is non-finite or outside the reader range.
    InvalidCoordinate {
        /// Latitude as `f64::to_bits()`.
        lat_deg_bits: u64,
        /// Longitude as `f64::to_bits()`.
        lon_deg_bits: u64,
    },
    /// Terrain tile index is outside the valid one-degree cell range.
    InvalidTileIndex {
        /// Latitude index.
        lat_index: i32,
        /// Longitude index.
        lon_index: i32,
    },
    /// Skadi tile identifier is malformed.
    InvalidTileId(String),
}

impl fmt::Display for DataCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCenter(center) => write!(f, "unknown analysis center {center:?}"),
            Self::UnknownProductType(product_type) => {
                write!(f, "unknown product type {product_type:?}")
            }
            Self::UnsupportedProduct {
                center,
                product_type,
            } => write!(f, "{center} does not serve {product_type}"),
            Self::NoOpenMirror {
                center,
                product_type,
            } => write!(f, "{center}/{product_type} has no open mirror"),
            Self::InvalidDate { year, month, day } => {
                write!(f, "invalid product date {year:04}-{month:02}-{day:02}")
            }
            Self::DateOutOfRange => write!(f, "product date is out of range"),
            Self::DateBeforeGpsEpoch(date) => {
                write!(f, "product date {date} is before the GPS week epoch")
            }
            Self::InvalidGpsDayOfWeek(day) => {
                write!(f, "invalid GPS day-of-week {day}")
            }
            Self::InvalidSample(sample) => write!(f, "invalid sample code {sample:?}"),
            Self::InvalidIssue(issue) => write!(f, "invalid issue time {issue:?}"),
            Self::MissingIssue { center } => write!(f, "{center} requires an issue time"),
            Self::UnexpectedIssue { center } => write!(f, "{center} does not take an issue time"),
            Self::UnsupportedIssue { center, issue } => {
                write!(f, "{center} does not publish issue {issue:?}")
            }
            Self::InvalidDateTime {
                hour,
                minute,
                second,
            } => write!(f, "invalid product time {hour:02}:{minute:02}:{second:02}"),
            Self::NoUltraIssue => write!(f, "no ultra-rapid issue at or before target"),
            Self::NoAvailableUltraIssue => {
                write!(f, "no available ultra-rapid issue at or before target")
            }
            Self::InvalidStation(station) => write!(f, "invalid station code {station:?}"),
            Self::InvalidCoordinate {
                lat_deg_bits,
                lon_deg_bits,
            } => write!(
                f,
                "invalid terrain coordinate lat={} lon={}",
                f64::from_bits(*lat_deg_bits),
                f64::from_bits(*lon_deg_bits)
            ),
            Self::InvalidTileIndex {
                lat_index,
                lon_index,
            } => write!(
                f,
                "invalid terrain tile index lat={lat_index} lon={lon_index}"
            ),
            Self::InvalidTileId(id) => write!(f, "invalid skadi tile id {id:?}"),
        }
    }
}

impl std::error::Error for DataCatalogError {}

/// Error returned by SRTM HGT to DTED conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HgtConversionError {
    /// The decompressed HGT payload is not the SRTM1 byte length.
    BadLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        got: usize,
    },
    /// Terrain tile index is outside the valid one-degree cell range.
    InvalidTileIndex {
        /// Latitude index.
        lat_index: i32,
        /// Longitude index.
        lon_index: i32,
    },
}

impl fmt::Display for HgtConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLength { expected, got } => {
                write!(
                    f,
                    "invalid SRTM1 HGT length: expected {expected}, got {got}"
                )
            }
            Self::InvalidTileIndex {
                lat_index,
                lon_index,
            } => write!(
                f,
                "invalid terrain tile index lat={lat_index} lon={lon_index}"
            ),
        }
    }
}

impl std::error::Error for HgtConversionError {}

const MIN_TERRAIN_LAT_INDEX: i32 = -90;
const MAX_TERRAIN_LAT_INDEX: i32 = 89;
const MIN_TERRAIN_LON_INDEX: i32 = -180;
const MAX_TERRAIN_LON_INDEX: i32 = 179;
const MIN_TERRAIN_LAT_DEG: f64 = -90.0;
const MAX_TERRAIN_LAT_DEG: f64 = 90.0;
const MIN_TERRAIN_LON_DEG: f64 = -180.0;
const MAX_TERRAIN_LON_DEG: f64 = 180.0;
const SRTM1_POSTINGS_PER_AXIS: usize = 3601;
const SRTM1_HGT_LEN: usize = SRTM1_POSTINGS_PER_AXIS * SRTM1_POSTINGS_PER_AXIS * 2;
const DTED_SRTM1_DATA_BLOCK_LEN: usize = 12 + 2 * SRTM1_POSTINGS_PER_AXIS;
const DTED_SRTM1_LEN: usize =
    terrain::DATA_OFFSET + SRTM1_POSTINGS_PER_AXIS * DTED_SRTM1_DATA_BLOCK_LEN;

/// Civil UTC date used by product archive names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProductDate {
    /// Year.
    pub year: i32,
    /// Month in `1..=12`.
    pub month: u8,
    /// Day of month.
    pub day: u8,
}

impl ProductDate {
    /// Build and validate a civil date.
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, DataCatalogError> {
        let days = days_in_month(i64::from(year), i64::from(month));
        if !(1..=9999).contains(&year) || days == 0 || day == 0 || i64::from(day) > days {
            return Err(DataCatalogError::InvalidDate { year, month, day });
        }
        Ok(Self { year, month, day })
    }

    /// Build a date from GPS week and day-of-week (`0` = Sunday).
    pub fn from_gps_week_day(week: u32, day_of_week: u8) -> Result<Self, DataCatalogError> {
        if day_of_week > 6 {
            return Err(DataCatalogError::InvalidGpsDayOfWeek(day_of_week));
        }
        let epoch_jdn =
            week_epoch_julian_day_number(TimeScale::Gpst).expect("GPST has a week-numbering epoch");
        let offset_days = i64::from(week)
            .checked_mul(7)
            .and_then(|days| days.checked_add(i64::from(day_of_week)))
            .ok_or(DataCatalogError::DateOutOfRange)?;
        product_date_from_jdn(
            epoch_jdn
                .checked_add(offset_days)
                .ok_or(DataCatalogError::DateOutOfRange)?,
        )
    }

    /// GPS week for this date.
    pub fn gps_week(self) -> Result<u32, DataCatalogError> {
        week_from_calendar(
            TimeScale::Gpst,
            i64::from(self.year),
            i64::from(self.month),
            i64::from(self.day),
        )
        .ok_or(DataCatalogError::DateBeforeGpsEpoch(self))
    }

    /// Day-of-year in `1..=366`.
    #[must_use]
    pub fn day_of_year(self) -> u16 {
        day_of_year_int(self.year, i32::from(self.month), i32::from(self.day)) as u16
    }

    fn add_days(self, days: i64) -> Result<Self, DataCatalogError> {
        product_date_from_jdn(
            self.julian_day_number()
                .checked_add(days)
                .ok_or(DataCatalogError::DateOutOfRange)?,
        )
    }

    fn julian_day_number(self) -> i64 {
        julian_day_number(self.year, i32::from(self.month), i32::from(self.day))
    }
}

impl fmt::Display for ProductDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Civil UTC date and time used for ultra-rapid issue selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProductDateTime {
    /// Date.
    pub date: ProductDate,
    /// Hour in `0..=23`.
    pub hour: u8,
    /// Minute in `0..=59`.
    pub minute: u8,
    /// Second in `0..=59`.
    pub second: u8,
}

impl ProductDateTime {
    /// Build and validate a civil date and time.
    pub fn new(
        date: ProductDate,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<Self, DataCatalogError> {
        if hour > 23 || minute > 59 || second > 59 {
            return Err(DataCatalogError::InvalidDateTime {
                hour,
                minute,
                second,
            });
        }
        Ok(Self {
            date,
            hour,
            minute,
            second,
        })
    }

    fn ordering_minutes(self) -> i64 {
        self.date.julian_day_number() * 1_440 + i64::from(self.hour) * 60 + i64::from(self.minute)
    }
}

/// Ultra-rapid issue date and `HHMM` issue time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UltraIssue {
    /// Product date.
    pub date: ProductDate,
    /// Issue time.
    pub issue: String,
}

impl UltraIssue {
    /// Build and validate an ultra-rapid issue.
    pub fn new(date: ProductDate, issue: &str) -> Result<Self, DataCatalogError> {
        validate_issue(issue)?;
        Ok(Self {
            date,
            issue: issue.to_string(),
        })
    }
}

/// One generated ultra-rapid SP3 archive candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraSp3Location {
    /// Stable catalog label identifying the primary, alternate, or alias rule.
    pub pattern: String,
    /// Product span token used by the candidate.
    pub span: String,
    /// Sampling token used by the candidate.
    pub sample: String,
    /// Archive filename without a transport compression suffix.
    pub filename: String,
    /// Full archive URL, including its compression suffix when applicable.
    pub url: String,
    /// Archive compression for this candidate.
    pub compression: ArchiveCompression,
}

#[derive(Debug, Clone, Copy)]
struct UltraSp3Pattern {
    label: &'static str,
    span: &'static str,
    sample: &'static str,
    alias_filename: Option<&'static str>,
}

const IGS_ULT_SP3_PATTERNS: [UltraSp3Pattern; 3] = [
    UltraSp3Pattern {
        label: "primary_02D_15M",
        span: "02D",
        sample: "15M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_02D_05M",
        span: "02D",
        sample: "05M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_01D_15M",
        span: "01D",
        sample: "15M",
        alias_filename: None,
    },
];

const COD_ULT_SP3_PATTERNS: [UltraSp3Pattern; 3] = [
    UltraSp3Pattern {
        label: "primary_01D_05M",
        span: "01D",
        sample: "05M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_02D_05M",
        span: "02D",
        sample: "05M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alias_latest",
        span: "01D",
        sample: "05M",
        alias_filename: Some("COD0OPSULT.SP3"),
    },
];

const ESA_ULT_SP3_PATTERNS: [UltraSp3Pattern; 3] = [
    UltraSp3Pattern {
        label: "primary_02D_05M",
        span: "02D",
        sample: "05M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_02D_15M",
        span: "02D",
        sample: "15M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_01D_05M",
        span: "01D",
        sample: "05M",
        alias_filename: None,
    },
];

const GFZ_ULT_SP3_PATTERNS: [UltraSp3Pattern; 3] = [
    UltraSp3Pattern {
        label: "primary_02D_05M",
        span: "02D",
        sample: "05M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_02D_15M",
        span: "02D",
        sample: "15M",
        alias_filename: None,
    },
    UltraSp3Pattern {
        label: "alternate_01D_05M",
        span: "01D",
        sample: "05M",
        alias_filename: None,
    },
];

/// A pure product specification that resolves to one archive filename and URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductSpec {
    /// Analysis center.
    pub center: AnalysisCenter,
    /// Product type.
    pub product_type: ProductType,
    /// Product date.
    pub date: ProductDate,
    /// Sampling token.
    pub sample: String,
    /// Optional issue time for ultra-rapid products.
    pub issue: Option<String>,
}

impl ProductSpec {
    /// Build a product specification and validate it against the catalog.
    pub fn new(
        center: AnalysisCenter,
        product_type: ProductType,
        date: ProductDate,
        sample: &str,
        issue: Option<&str>,
    ) -> Result<Self, DataCatalogError> {
        validate_product(center, product_type, sample, issue)?;
        Ok(Self {
            center,
            product_type,
            date,
            sample: sample.to_string(),
            issue: issue.map(ToOwned::to_owned),
        })
    }

    /// GPS week for the product date.
    pub fn gps_week(&self) -> Result<u32, DataCatalogError> {
        self.date.gps_week()
    }

    /// Day-of-year for the product date.
    #[must_use]
    pub fn day_of_year(&self) -> u16 {
        self.date.day_of_year()
    }

    /// Canonical IGS long-name filename without archive compression suffix.
    pub fn canonical_filename(&self) -> Result<String, DataCatalogError> {
        let convention = validate_product(
            self.center,
            self.product_type,
            &self.sample,
            self.issue.as_deref(),
        )?;
        let descriptor = product_type_convention(self.product_type);
        Ok(match descriptor.kind {
            ProductFilenameKind::Sampled => format!(
                "{}_{}_{}_{}_{}.{}",
                convention.token,
                date_block(self.date, self.issue.as_deref()),
                convention.span,
                self.sample,
                descriptor.content_code,
                descriptor.extension
            ),
            ProductFilenameKind::Nav => format!(
                "{}_R_{}_{}_{}.{}",
                convention.token,
                date_block(self.date, None),
                convention.span,
                descriptor.content_code,
                descriptor.extension
            ),
        })
    }

    /// Full archive URL, including `.gz` when the cataloged archive is gzipped.
    pub fn archive_url(&self) -> Result<String, DataCatalogError> {
        let convention = validate_product(
            self.center,
            self.product_type,
            &self.sample,
            self.issue.as_deref(),
        )?;
        let entry = center_catalog(self.center).expect("catalog entry exists for enum variant");
        let filename = self.canonical_filename()?;
        Ok(format!(
            "{}/{}/{}{}",
            entry.root_url,
            dir_path(convention.layout, self.date)?,
            filename,
            convention.compression.suffix()
        ))
    }
}

/// A pure station observation specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StationObservationSpec {
    /// 9-character RINEX 3 site identifier.
    pub station: String,
    /// Observation date.
    pub date: ProductDate,
    /// Sampling token.
    pub sample: String,
}

impl StationObservationSpec {
    /// Build and validate a daily station observation product.
    pub fn new(station: &str, date: ProductDate, sample: &str) -> Result<Self, DataCatalogError> {
        validate_station(station)?;
        validate_sample(sample)?;
        Ok(Self {
            station: station.to_string(),
            date,
            sample: sample.to_string(),
        })
    }

    /// Canonical RINEX 3 CRINEX filename without archive compression suffix.
    pub fn canonical_filename(&self) -> Result<String, DataCatalogError> {
        station_obs_filename(&self.station, self.date, &self.sample)
    }

    /// Full archive URL, including `.gz`.
    pub fn archive_url(&self) -> Result<String, DataCatalogError> {
        station_obs_url(&self.station, self.date, &self.sample)
    }
}

/// Static catalog entries, in the same order as the binding data catalog.
#[must_use]
pub const fn catalog() -> &'static [CenterCatalogEntry] {
    &CATALOG
}

/// Supported center codes, in catalog order.
#[must_use]
pub const fn centers() -> &'static [AnalysisCenter] {
    &CENTER_ORDER
}

/// Supported product types.
#[must_use]
pub const fn product_types() -> &'static [ProductTypeConvention] {
    &PRODUCT_TYPE_CONVENTIONS
}

/// Archive hosts present in the catalog.
#[must_use]
pub const fn allowed_hosts() -> &'static [&'static str] {
    &ALLOWED_HOSTS
}

/// Catalog entry for the Skadi SRTM terrain source.
#[must_use]
pub const fn skadi_source_entry() -> TerrainSourceEntry {
    SKADI_SOURCE
}

/// Catalog entry for the CelesTrak CSSI space-weather source.
#[must_use]
pub const fn space_weather_source_entry() -> SpaceWeatherSourceEntry {
    CELESTRAK_SPACE_WEATHER_SOURCE
}

/// Filename for a CelesTrak space-weather product.
#[must_use]
pub const fn space_weather_filename(product: SpaceWeatherProduct) -> &'static str {
    match product {
        SpaceWeatherProduct::All => "SW-All.csv",
        SpaceWeatherProduct::Last5Years => "SW-Last5Years.csv",
    }
}

/// Build the CelesTrak archive URL for a space-weather product.
#[must_use]
pub fn space_weather_archive_url(product: SpaceWeatherProduct) -> String {
    format!(
        "{}/{}",
        CELESTRAK_SPACE_WEATHER_SOURCE.root_url,
        space_weather_filename(product)
    )
}

/// Build the cache relative path for a space-weather product.
#[must_use]
pub fn space_weather_cache_relpath(product: SpaceWeatherProduct) -> String {
    format!("space-weather/{}", space_weather_filename(product))
}

/// Build the Skadi SRTM tile id, for example `N36W107`.
pub fn skadi_tile_id(lat_index: i32, lon_index: i32) -> Result<String, DataCatalogError> {
    validate_terrain_tile_index(lat_index, lon_index)?;
    let lat_hemi = if lat_index >= 0 { 'N' } else { 'S' };
    let lon_hemi = if lon_index >= 0 { 'E' } else { 'W' };
    Ok(format!(
        "{lat_hemi}{:02}{lon_hemi}{:03}",
        lat_index.abs(),
        lon_index.abs()
    ))
}

/// Build the Skadi latitude band directory, for example `N36`.
pub fn skadi_band(lat_index: i32) -> Result<String, DataCatalogError> {
    validate_terrain_lat_index(lat_index)?;
    let lat_hemi = if lat_index >= 0 { 'N' } else { 'S' };
    Ok(format!("{lat_hemi}{:02}", lat_index.abs()))
}

/// Build the Skadi SRTM archive URL for a tile.
pub fn skadi_archive_url(lat_index: i32, lon_index: i32) -> Result<String, DataCatalogError> {
    let band = skadi_band(lat_index)?;
    let tile_id = skadi_tile_id(lat_index, lon_index)?;
    Ok(format!(
        "{}/skadi/{}/{}.hgt{}",
        SKADI_SOURCE.root_url,
        band,
        tile_id,
        SKADI_SOURCE.compression.suffix()
    ))
}

/// Build the DTED tile filename read by the terrain module.
pub fn dted_tile_filename(lat_index: i32, lon_index: i32) -> Result<String, DataCatalogError> {
    validate_terrain_tile_index(lat_index, lon_index)?;
    Ok(format!(
        "{}_{}{}",
        terrain::format_lat(lat_index),
        terrain::format_lon(lon_index),
        terrain::DTED_SUFFIX
    ))
}

/// Build the DTED ten-degree cache block directory read by the terrain module.
pub fn dted_block_dir(lat_index: i32, lon_index: i32) -> Result<String, DataCatalogError> {
    validate_terrain_tile_index(lat_index, lon_index)?;
    Ok(terrain::terrain_block_dir(lat_index, lon_index))
}

/// Build the DTED cache relative path read by the terrain module.
pub fn dted_cache_relpath(lat_index: i32, lon_index: i32) -> Result<String, DataCatalogError> {
    Ok(format!(
        "{}/{}",
        dted_block_dir(lat_index, lon_index)?,
        dted_tile_filename(lat_index, lon_index)?
    ))
}

/// Parse a Skadi SRTM tile id into `(lat_index, lon_index)`.
pub fn parse_skadi_tile_id(id: &str) -> Result<(i32, i32), DataCatalogError> {
    let bytes = id.as_bytes();
    if bytes.len() != 7
        || !matches!(bytes[0], b'N' | b'S')
        || !matches!(bytes[3], b'E' | b'W')
        || !bytes[1..3].iter().all(u8::is_ascii_digit)
        || !bytes[4..7].iter().all(u8::is_ascii_digit)
    {
        return Err(DataCatalogError::InvalidTileId(id.to_string()));
    }

    let lat_abs = id[1..3]
        .parse::<i32>()
        .map_err(|_| DataCatalogError::InvalidTileId(id.to_string()))?;
    let lon_abs = id[4..7]
        .parse::<i32>()
        .map_err(|_| DataCatalogError::InvalidTileId(id.to_string()))?;
    if (bytes[0] == b'S' && lat_abs == 0) || (bytes[3] == b'W' && lon_abs == 0) {
        return Err(DataCatalogError::InvalidTileId(id.to_string()));
    }

    let lat_index = if bytes[0] == b'N' { lat_abs } else { -lat_abs };
    let lon_index = if bytes[3] == b'E' { lon_abs } else { -lon_abs };
    validate_terrain_tile_index(lat_index, lon_index)?;
    Ok((lat_index, lon_index))
}

/// Derive the terrain tile index covering a latitude/longitude coordinate.
pub fn terrain_tile_index(lat_deg: f64, lon_deg: f64) -> Result<(i32, i32), DataCatalogError> {
    if !lat_deg.is_finite()
        || !lon_deg.is_finite()
        || !(MIN_TERRAIN_LAT_DEG..=MAX_TERRAIN_LAT_DEG).contains(&lat_deg)
        || !(MIN_TERRAIN_LON_DEG..=MAX_TERRAIN_LON_DEG).contains(&lon_deg)
    {
        return Err(DataCatalogError::InvalidCoordinate {
            lat_deg_bits: lat_deg.to_bits(),
            lon_deg_bits: lon_deg.to_bits(),
        });
    }

    let (mut lat_index, mut lon_index) = terrain::terrain_grid(lon_deg, lat_deg);
    if lat_index == MAX_TERRAIN_LAT_DEG as i32 {
        lat_index = MAX_TERRAIN_LAT_INDEX;
    }
    if lon_index == MAX_TERRAIN_LON_DEG as i32 {
        lon_index = MAX_TERRAIN_LON_INDEX;
    }
    validate_terrain_tile_index(lat_index, lon_index)?;
    Ok((lat_index, lon_index))
}

/// Convert decompressed SRTM1 HGT bytes into deterministic DTED `.dt2` bytes.
///
/// The HGT payload must be 3601 by 3601 big-endian `i16` samples in row-major
/// order. HGT rows run north to south; DTED data records are longitude columns
/// with postings south to north, so output posting `(i, j)` reads source sample
/// `hgt[r = 3600 - i][c = j]`. SRTM void samples (`-32768`) are written as sea
/// level (`0`) so the existing terrain reader returns `0` for those postings.
pub fn hgt_to_dted(
    lat_index: i32,
    lon_index: i32,
    hgt: &[u8],
) -> Result<Vec<u8>, HgtConversionError> {
    validate_hgt_tile_index(lat_index, lon_index)?;
    if hgt.len() != SRTM1_HGT_LEN {
        return Err(HgtConversionError::BadLength {
            expected: SRTM1_HGT_LEN,
            got: hgt.len(),
        });
    }

    let mut out = vec![b' '; DTED_SRTM1_LEN];
    out[0..4].copy_from_slice(b"UHL1");
    out[4..12].copy_from_slice(dted_coord_field(lon_index, true).as_bytes());
    out[12..20].copy_from_slice(dted_coord_field(lat_index, false).as_bytes());
    out[47..51].copy_from_slice(b"3601");
    out[51..55].copy_from_slice(b"3601");

    for lon_posting in 0..SRTM1_POSTINGS_PER_AXIS {
        let block_start = terrain::DATA_OFFSET + lon_posting * DTED_SRTM1_DATA_BLOCK_LEN;
        let checksum_start = block_start + DTED_SRTM1_DATA_BLOCK_LEN - 4;
        out[block_start] = terrain::DATA_SENTINEL;

        let count = (lon_posting as u32).to_be_bytes();
        out[block_start + 1..block_start + 4].copy_from_slice(&count[1..4]);
        out[block_start + 4..block_start + 6].copy_from_slice(&(lon_posting as u16).to_be_bytes());
        out[block_start + 6..block_start + 8].copy_from_slice(&0u16.to_be_bytes());

        for lat_posting in 0..SRTM1_POSTINGS_PER_AXIS {
            let hgt_row = SRTM1_POSTINGS_PER_AXIS - 1 - lat_posting;
            let hgt_sample_start = 2 * (hgt_row * SRTM1_POSTINGS_PER_AXIS + lon_posting);
            let sample = i16::from_be_bytes([hgt[hgt_sample_start], hgt[hgt_sample_start + 1]]);
            let encoded = encode_dted_signed_magnitude(sample).to_be_bytes();
            let dted_sample_start = block_start + 8 + 2 * lat_posting;
            out[dted_sample_start..dted_sample_start + 2].copy_from_slice(&encoded);
        }

        let checksum = out[block_start..checksum_start]
            .iter()
            .fold(0i32, |acc, byte| acc + i32::from(*byte));
        out[checksum_start..checksum_start + 4].copy_from_slice(&checksum.to_be_bytes());
    }

    debug_assert_eq!(out.len(), 25_981_042);
    Ok(out)
}

/// Product pairs intentionally withheld because no open mirror is known.
#[must_use]
pub const fn no_open_mirrors() -> &'static [NoOpenMirrorProduct] {
    &NO_OPEN_MIRRORS
}

/// Confirm that a center/product pair has an open catalog mirror.
pub fn open_mirror(
    center: AnalysisCenter,
    product_type: ProductType,
) -> Result<(), DataCatalogError> {
    open_mirror_code(center.code(), product_type.code())
}

/// Confirm that a center/product code pair is not in the no-open-mirror list.
pub fn open_mirror_code(center: &str, product_type: &str) -> Result<(), DataCatalogError> {
    if NO_OPEN_MIRRORS
        .iter()
        .any(|entry| entry.center == center && entry.product_type == product_type)
    {
        Err(DataCatalogError::NoOpenMirror {
            center: center.to_string(),
            product_type: product_type.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Look up a center's static catalog entry.
#[must_use]
pub fn center_catalog(center: AnalysisCenter) -> Option<&'static CenterCatalogEntry> {
    CATALOG.iter().find(|entry| entry.center == center)
}

/// Look up the convention for one center and product type.
pub fn product_convention(
    center: AnalysisCenter,
    product_type: ProductType,
) -> Result<&'static CenterProductConvention, DataCatalogError> {
    open_mirror(center, product_type)?;
    let entry = center_catalog(center).expect("catalog entry exists for enum variant");
    entry
        .products
        .iter()
        .find(|product| product.product_type == product_type)
        .ok_or(DataCatalogError::UnsupportedProduct {
            center,
            product_type,
        })
}

/// Default sampling token for a center/product pair.
pub fn default_sample(
    center: AnalysisCenter,
    product_type: ProductType,
) -> Result<&'static str, DataCatalogError> {
    Ok(product_convention(center, product_type)?.default_sample)
}

/// GPS week number for a product date.
pub fn gps_week(date: ProductDate) -> Result<u32, DataCatalogError> {
    date.gps_week()
}

/// Day-of-year in `1..=366` for a product date.
#[must_use]
pub fn day_of_year(date: ProductDate) -> u16 {
    date.day_of_year()
}

/// Build a product specification for any center/product/date combination.
pub fn product(
    center: AnalysisCenter,
    product_type: ProductType,
    date: ProductDate,
    sample: Option<&str>,
    issue: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    let sample = match sample {
        Some(sample) => sample,
        None => default_sample(center, product_type)?,
    };
    ProductSpec::new(center, product_type, date, sample, issue)
}

/// Build the canonical IGS long-name filename for a product.
pub fn canonical_filename(
    center: AnalysisCenter,
    product_type: ProductType,
    date: ProductDate,
    sample: Option<&str>,
    issue: Option<&str>,
) -> Result<String, DataCatalogError> {
    product(center, product_type, date, sample, issue)?.canonical_filename()
}

/// Build the full archive URL for a product.
pub fn archive_url(
    center: AnalysisCenter,
    product_type: ProductType,
    date: ProductDate,
    sample: Option<&str>,
    issue: Option<&str>,
) -> Result<String, DataCatalogError> {
    product(center, product_type, date, sample, issue)?.archive_url()
}

/// Build a clock product for a center and date.
pub fn mgex_clk(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    product(center, ProductType::Clk, date, sample, None)
}

/// Build a merged broadcast-navigation product for a center and date.
pub fn mgex_nav(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    product(center, ProductType::Nav, date, sample, None)
}

/// Build an IONEX product for a center and date.
pub fn mgex_ionex(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    product(center, ProductType::Ionex, date, sample, None)
}

/// Build the CODE rapid IONEX product for a date.
pub fn rapid_ionex(
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    product(
        AnalysisCenter::CodRap,
        ProductType::Ionex,
        date,
        sample,
        None,
    )
}

/// Day offset for predicted IONEX aliases.
#[must_use]
pub const fn predicted_day_offset(center: AnalysisCenter) -> i64 {
    match center {
        AnalysisCenter::CodPrd2 => 1,
        _ => 0,
    }
}

/// Build a CODE predicted IONEX product for a target date.
pub fn predicted_ionex(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    match center {
        AnalysisCenter::CodPrd1 | AnalysisCenter::CodPrd2 => {
            let target = date.add_days(predicted_day_offset(center))?;
            product(center, ProductType::Ionex, target, sample, None)
        }
        other => Err(DataCatalogError::UnsupportedProduct {
            center: other,
            product_type: ProductType::Ionex,
        }),
    }
}

/// Build an SP3 product for a center and date.
pub fn mgex_sp3(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    product(center, ProductType::Sp3, date, sample, None)
}

/// Build an ultra-rapid OPS SP3 product for a date and issue time.
pub fn ops_ultra_sp3(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
    issue: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    let issue = issue.unwrap_or("0000");
    product(center, ProductType::Sp3, date, sample, Some(issue))
}

/// Generate the current primary ultra-rapid SP3 location followed by known
/// duration/sampling alternates and documented latest-product aliases.
///
/// Candidate order is deterministic and center-specific. Callers should try
/// the next location only when the prior archive URL is absent; transport and
/// retry policy remain outside the pure core catalog.
pub fn ultra_sp3_locations(
    center: AnalysisCenter,
    date: ProductDate,
    issue: &str,
) -> Result<Vec<UltraSp3Location>, DataCatalogError> {
    validate_issue_for_center(center, Some(issue))?;
    let patterns: &[UltraSp3Pattern] = match center {
        AnalysisCenter::IgsUlt => &IGS_ULT_SP3_PATTERNS,
        AnalysisCenter::CodUlt => &COD_ULT_SP3_PATTERNS,
        AnalysisCenter::EsaUlt => &ESA_ULT_SP3_PATTERNS,
        AnalysisCenter::GfzUlt => &GFZ_ULT_SP3_PATTERNS,
        other => {
            return Err(DataCatalogError::UnsupportedProduct {
                center: other,
                product_type: ProductType::Sp3,
            })
        }
    };
    let convention = product_convention(center, ProductType::Sp3)?;
    let entry = center_catalog(center).expect("catalog entry exists for enum variant");
    let directory = dir_path(convention.layout, date)?;
    let date = date_block(date, Some(issue));

    Ok(patterns
        .iter()
        .map(|pattern| {
            let filename = pattern.alias_filename.map_or_else(
                || {
                    format!(
                        "{}_{}_{}_{}_ORB.SP3",
                        convention.token, date, pattern.span, pattern.sample
                    )
                },
                ToOwned::to_owned,
            );
            UltraSp3Location {
                pattern: pattern.label.to_string(),
                span: pattern.span.to_string(),
                sample: pattern.sample.to_string(),
                url: format!(
                    "{}/{}/{}{}",
                    entry.root_url,
                    directory,
                    filename,
                    convention.compression.suffix()
                ),
                filename,
                compression: convention.compression,
            }
        })
        .collect())
}

/// Build an ultra-rapid OPS clock product for a date and issue time.
pub fn ops_ultra_clk(
    center: AnalysisCenter,
    date: ProductDate,
    sample: Option<&str>,
    issue: Option<&str>,
) -> Result<ProductSpec, DataCatalogError> {
    let issue = issue.unwrap_or("0000");
    product(center, ProductType::Clk, date, sample, Some(issue))
}

/// Select the latest ultra-rapid OPS SP3 issue at or before a target time.
pub fn latest_ops_ultra_sp3(
    center: AnalysisCenter,
    target: ProductDateTime,
    sample: Option<&str>,
    available_issues: Option<&[UltraIssue]>,
) -> Result<ProductSpec, DataCatalogError> {
    let selected = latest_ultra_issue(center, target, available_issues)?;
    ops_ultra_sp3(center, selected.date, sample, Some(&selected.issue))
}

/// Candidate ultra-rapid issues at or before a target time, newest first.
pub fn ultra_issue_candidates(
    center: AnalysisCenter,
    target: ProductDateTime,
) -> Result<Vec<UltraIssue>, DataCatalogError> {
    let entry = center_catalog(center).expect("catalog entry exists for enum variant");
    let _ = product_convention(center, ProductType::Sp3)?;
    if entry.issues.is_empty() {
        return Err(DataCatalogError::UnsupportedProduct {
            center,
            product_type: ProductType::Sp3,
        });
    }

    let mut candidates = Vec::new();
    for date in [target.date, target.date.add_days(-1)?] {
        for issue in entry.issues.iter().rev() {
            if issue_ordering_minutes(date, issue)? <= target.ordering_minutes() {
                candidates.push(UltraIssue::new(date, issue)?);
            }
        }
    }
    Ok(candidates)
}

/// Latest ultra-rapid issue at or before a target time.
pub fn latest_ultra_issue(
    center: AnalysisCenter,
    target: ProductDateTime,
    available_issues: Option<&[UltraIssue]>,
) -> Result<UltraIssue, DataCatalogError> {
    let candidates = ultra_issue_candidates(center, target)?;
    if candidates.is_empty() {
        return Err(DataCatalogError::NoUltraIssue);
    }
    if let Some(available) = available_issues {
        candidates
            .into_iter()
            .find(|candidate| {
                available
                    .iter()
                    .any(|issue| issue.date == candidate.date && issue.issue == candidate.issue)
            })
            .ok_or(DataCatalogError::NoAvailableUltraIssue)
    } else {
        Ok(candidates[0].clone())
    }
}

/// Candidate IONEX dates at or before a target date, newest first.
pub fn gim_date_candidates(
    center: AnalysisCenter,
    target: ProductDate,
    lookback: u32,
) -> Result<Vec<ProductDate>, DataCatalogError> {
    let _ = product_convention(center, ProductType::Ionex)?;
    let base = target.add_days(predicted_day_offset(center))?;
    let mut out = Vec::with_capacity(usize::try_from(lookback).unwrap_or(usize::MAX));
    for back in 0..=lookback {
        out.push(base.add_days(-i64::from(back))?);
    }
    Ok(out)
}

/// Build a daily station observation product.
pub fn station_obs(
    station: &str,
    date: ProductDate,
    sample: Option<&str>,
) -> Result<StationObservationSpec, DataCatalogError> {
    StationObservationSpec::new(station, date, sample.unwrap_or("30S"))
}

/// Build the canonical RINEX 3 CRINEX filename for a daily station observation.
pub fn station_obs_filename(
    station: &str,
    date: ProductDate,
    sample: &str,
) -> Result<String, DataCatalogError> {
    validate_station(station)?;
    validate_sample(sample)?;
    Ok(format!(
        "{}_R_{}_01D_{}_MO.crx",
        station,
        date_block(date, None),
        sample
    ))
}

/// Build the full BKG IGS archive URL for a daily station observation.
pub fn station_obs_url(
    station: &str,
    date: ProductDate,
    sample: &str,
) -> Result<String, DataCatalogError> {
    let filename = station_obs_filename(station, date, sample)?;
    Ok(format!(
        "https://igs.bkg.bund.de/root_ftp/IGS/{}/{}.gz",
        dir_path(ArchiveLayout::BkgObsYearDoy, date)?,
        filename
    ))
}

/// The transfer protocol for the daily station observation archive.
#[must_use]
pub const fn station_obs_protocol() -> ArchiveProtocol {
    ArchiveProtocol::Https
}

fn validate_terrain_lat_index(lat_index: i32) -> Result<(), DataCatalogError> {
    if (MIN_TERRAIN_LAT_INDEX..=MAX_TERRAIN_LAT_INDEX).contains(&lat_index) {
        Ok(())
    } else {
        Err(DataCatalogError::InvalidTileIndex {
            lat_index,
            lon_index: 0,
        })
    }
}

fn validate_terrain_tile_index(lat_index: i32, lon_index: i32) -> Result<(), DataCatalogError> {
    if (MIN_TERRAIN_LAT_INDEX..=MAX_TERRAIN_LAT_INDEX).contains(&lat_index)
        && (MIN_TERRAIN_LON_INDEX..=MAX_TERRAIN_LON_INDEX).contains(&lon_index)
    {
        Ok(())
    } else {
        Err(DataCatalogError::InvalidTileIndex {
            lat_index,
            lon_index,
        })
    }
}

fn validate_hgt_tile_index(lat_index: i32, lon_index: i32) -> Result<(), HgtConversionError> {
    if (MIN_TERRAIN_LAT_INDEX..=MAX_TERRAIN_LAT_INDEX).contains(&lat_index)
        && (MIN_TERRAIN_LON_INDEX..=MAX_TERRAIN_LON_INDEX).contains(&lon_index)
    {
        Ok(())
    } else {
        Err(HgtConversionError::InvalidTileIndex {
            lat_index,
            lon_index,
        })
    }
}

fn dted_coord_field(index: i32, is_longitude: bool) -> String {
    let hemi = match (is_longitude, index >= 0) {
        (true, true) => 'E',
        (true, false) => 'W',
        (false, true) => 'N',
        (false, false) => 'S',
    };
    format!("{:03}0000{hemi}", index.abs())
}

fn encode_dted_signed_magnitude(sample: i16) -> u16 {
    if sample == i16::MIN {
        0
    } else if sample >= 0 {
        sample as u16
    } else {
        0x8000 | (-i32::from(sample) as u16)
    }
}

fn product_type_convention(product_type: ProductType) -> &'static ProductTypeConvention {
    PRODUCT_TYPE_CONVENTIONS
        .iter()
        .find(|descriptor| descriptor.product_type == product_type)
        .expect("product descriptor exists for enum variant")
}

fn validate_product(
    center: AnalysisCenter,
    product_type: ProductType,
    sample: &str,
    issue: Option<&str>,
) -> Result<&'static CenterProductConvention, DataCatalogError> {
    let convention = product_convention(center, product_type)?;
    validate_sample(sample)?;
    validate_issue_for_center(center, issue)?;
    Ok(convention)
}

fn validate_issue_for_center(
    center: AnalysisCenter,
    issue: Option<&str>,
) -> Result<(), DataCatalogError> {
    let entry = center_catalog(center).expect("catalog entry exists for enum variant");
    match (entry.issues.is_empty(), issue) {
        (true, None) => Ok(()),
        (true, Some(_)) => Err(DataCatalogError::UnexpectedIssue { center }),
        (false, None) => Err(DataCatalogError::MissingIssue { center }),
        (false, Some(issue)) => {
            validate_issue(issue)?;
            if entry.issues.contains(&issue) {
                Ok(())
            } else {
                Err(DataCatalogError::UnsupportedIssue {
                    center,
                    issue: issue.to_string(),
                })
            }
        }
    }
}

fn validate_sample(sample: &str) -> Result<(), DataCatalogError> {
    let bytes = sample.as_bytes();
    let valid = bytes.len() == 3
        && bytes[0].is_ascii_digit()
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_uppercase();
    if valid {
        Ok(())
    } else {
        Err(DataCatalogError::InvalidSample(sample.to_string()))
    }
}

fn validate_issue(issue: &str) -> Result<(), DataCatalogError> {
    let bytes = issue.as_bytes();
    let valid_digits = bytes.len() == 4 && bytes.iter().all(u8::is_ascii_digit);
    if !valid_digits {
        return Err(DataCatalogError::InvalidIssue(issue.to_string()));
    }
    let hour = issue[0..2]
        .parse::<u8>()
        .map_err(|_| DataCatalogError::InvalidIssue(issue.to_string()))?;
    let minute = issue[2..4]
        .parse::<u8>()
        .map_err(|_| DataCatalogError::InvalidIssue(issue.to_string()))?;
    if hour <= 23 && minute <= 59 {
        Ok(())
    } else {
        Err(DataCatalogError::InvalidIssue(issue.to_string()))
    }
}

fn validate_station(station: &str) -> Result<(), DataCatalogError> {
    let bytes = station.as_bytes();
    let valid = bytes.len() == 9
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit());
    if valid {
        Ok(())
    } else {
        Err(DataCatalogError::InvalidStation(station.to_string()))
    }
}

fn issue_minutes(issue: &str) -> Result<u16, DataCatalogError> {
    validate_issue(issue)?;
    let hour = issue[0..2]
        .parse::<u16>()
        .map_err(|_| DataCatalogError::InvalidIssue(issue.to_string()))?;
    let minute = issue[2..4]
        .parse::<u16>()
        .map_err(|_| DataCatalogError::InvalidIssue(issue.to_string()))?;
    Ok(hour * 60 + minute)
}

fn issue_ordering_minutes(date: ProductDate, issue: &str) -> Result<i64, DataCatalogError> {
    Ok(date.julian_day_number() * 1_440 + i64::from(issue_minutes(issue)?))
}

fn date_block(date: ProductDate, issue: Option<&str>) -> String {
    format!(
        "{}{:03}{}",
        date.year,
        date.day_of_year(),
        issue.unwrap_or("0000")
    )
}

fn dir_path(layout: ArchiveLayout, date: ProductDate) -> Result<String, DataCatalogError> {
    Ok(match layout {
        ArchiveLayout::GfzRapidWeek => format!("rapid/w{}", date.gps_week()?),
        ArchiveLayout::GfzUltraWeek => format!("ultra/w{}", date.gps_week()?),
        ArchiveLayout::GpsWeek => date.gps_week()?.to_string(),
        ArchiveLayout::BkgProductsWeek => format!("products/{}", date.gps_week()?),
        ArchiveLayout::BkgBrdcYearDoy => {
            format!("BRDC/{}/{:03}", date.year, date.day_of_year())
        }
        ArchiveLayout::BkgObsYearDoy => format!("obs/{}/{:03}", date.year, date.day_of_year()),
        ArchiveLayout::AiubCodeMgexYear => format!("CODE_MGEX/CODE/{}", date.year),
        ArchiveLayout::AiubCodeYear => format!("CODE/{}", date.year),
        ArchiveLayout::AiubCodeRoot => "CODE".to_string(),
    })
}

fn product_date_from_jdn(jdn: i64) -> Result<ProductDate, DataCatalogError> {
    let (year, month, day) = civil_from_julian_day_number(jdn);
    let year = i32::try_from(year).map_err(|_| DataCatalogError::DateOutOfRange)?;
    let month = u8::try_from(month).map_err(|_| DataCatalogError::DateOutOfRange)?;
    let day = u8::try_from(day).map_err(|_| DataCatalogError::DateOutOfRange)?;
    ProductDate::new(year, month, day).map_err(|_| DataCatalogError::DateOutOfRange)
}
