use sidereon_core::ephemeris::{Sp3, Sp3TimeSystem};

const SP3_WITH_GALILEO_TIME: &str = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GAL ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* PUBLIC API TEST FIXTURE
*  2020  6 24  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    123.456789
EOF
";

fn public_time_system_label(time_system: Sp3TimeSystem) -> &'static str {
    match time_system {
        Sp3TimeSystem::Gps => "GPS",
        Sp3TimeSystem::Glonass => "GLO",
        Sp3TimeSystem::Galileo => "GAL",
        Sp3TimeSystem::Tai => "TAI",
        Sp3TimeSystem::Utc => "UTC",
        Sp3TimeSystem::Qzss => "QZS",
        Sp3TimeSystem::Beidou => "BDT",
        Sp3TimeSystem::Irnss => "IRN",
    }
}

#[test]
fn ephemeris_reexports_sp3_time_system_variants() {
    let sp3 = Sp3::parse(SP3_WITH_GALILEO_TIME.as_bytes()).expect("parse SP3 fixture");

    assert_eq!(
        public_time_system_label(sp3.header.time_system),
        "GAL",
        "downstream users can name and match Sp3TimeSystem variants"
    );
}
