#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::atmosphere::Ionex;

fuzz_target!(|data: &[u8]| {
    parsed_product_round_trips(data);
    mixed_unit_product_round_trips(data);
});

/// Whatever product the bytes read as writes back and reads as itself.
fn parsed_product_round_trips(data: &[u8]) {
    let Ok(original) = Ionex::parse(data) else {
        return;
    };
    // The writer refuses a product with a value or field its IONEX field cannot
    // hold exactly; there is nothing to round-trip then.
    let Ok(encoded) = original.to_ionex_string() else {
        return;
    };
    assert_rereads_as(&original, &encoded);
}

/// A product whose values are whole numbers of different powers of ten, across
/// bands and, after nodes are swapped between bands, within them: the writer
/// writes it exactly, with an EXPONENT record per data block and split bands
/// where one file exponent does not.
fn mixed_unit_product_round_trips(data: &[u8]) {
    let mut bytes = data.iter().copied();
    let mut next = move || bytes.next().unwrap_or(0);
    let exponent = |byte: u8| i32::from(byte % 11) - 6;

    let mut text = String::new();
    for (data, label) in [
        ("     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"),
        ("  2020     1     1     0     0     0", "EPOCH OF FIRST MAP"),
        ("  2020     1     1     0     0     0", "EPOCH OF LAST MAP"),
        ("     0", "INTERVAL"),
        ("     1", "# OF MAPS IN FILE"),
        ("  COSZ", "MAPPING FUNCTION"),
        ("     0.0", "ELEVATION CUTOFF"),
        ("", "OBSERVABLES USED"),
        ("  6371.0", "BASE RADIUS"),
        ("     2", "MAP DIMENSION"),
        ("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        ("     1.0   0.0  -1.0", "LAT1 / LAT2 / DLAT"),
        ("     0.0   3.0   1.0", "LON1 / LON2 / DLON"),
    ] {
        record(&mut text, data, label);
    }
    record(&mut text, &format!("{:6}", exponent(next())), "EXPONENT");
    record(&mut text, "", "END OF HEADER");
    record(&mut text, "     1", "START OF TEC MAP");
    record(&mut text, "  2020     1     1     0     0     0", "EPOCH OF CURRENT MAP");
    for lat in ["   1.0", "   0.0"] {
        record(&mut text, &format!("{:6}", exponent(next())), "EXPONENT");
        record(
            &mut text,
            &format!("  {lat}   0.0   3.0   1.0 450.0"),
            "LAT/LON1/LON2/DLON/H",
        );
        for _ in 0..4 {
            let raw = i64::from(u16::from_le_bytes([next(), next()]) % 20_000) - 9_999;
            text.push_str(&format!("{raw:5}"));
        }
        text.push('\n');
    }
    record(&mut text, "     1", "END OF TEC MAP");
    record(&mut text, "", "END OF FILE");

    let product = Ionex::parse_str(&text).expect("a well-formed mixed-unit product reads");
    let mut samples = product.tec_grid_samples();
    for lon in 0..4 {
        if next() & 1 == 1 {
            let north = samples.tec_maps[0][0][lon];
            samples.tec_maps[0][0][lon] = samples.tec_maps[0][1][lon];
            samples.tec_maps[0][1][lon] = north;
        }
    }
    let product = Ionex::from_samples(samples).expect("swapped nodes rebuild a product");
    let encoded = product
        .to_ionex_string()
        .expect("each value is a whole number of some unit within I5");
    assert_rereads_as(&product, &encoded);

    // Arbitrary decimals at the nodes, which is what a product built from
    // samples rather than read from a file holds. A hundredth is a whole number
    // of 10^-2 units, so the writer writes each one exactly, though no value is
    // a whole number of the units the file above was written in.
    let mut samples = product.tec_grid_samples();
    for lat in 0..2 {
        for lon in 0..4 {
            let raw = i64::from(u16::from_le_bytes([next(), next()]) % 20_000) - 10_000;
            samples.tec_maps[0][lat][lon] = Some(raw as f64 / 100.0);
        }
    }
    let decimals = Ionex::from_samples(samples).expect("decimal samples rebuild a product");
    let encoded = decimals
        .to_ionex_string()
        .expect("a hundredth is a field the file states");
    assert_rereads_within_one_ulp(&decimals, &encoded);
}

/// As `assert_rereads_as`, for values built from samples rather than read from a
/// file: every node within one unit in the last place.
///
/// The writer states a value with the field the file can hold it in, and the
/// reader forms the product the reference readers form, so a value that never
/// came from a field can come back one unit away.
fn assert_rereads_within_one_ulp(original: &Ionex, encoded: &str) {
    let reparsed = Ionex::parse(encoded.as_bytes()).expect("encoded IONEX must reparse");
    let maps = original.tec_maps();
    let rebuilt = reparsed.tec_maps();
    assert_eq!(maps.len(), rebuilt.len());
    for (map, remap) in maps.iter().zip(rebuilt) {
        for (row, rerow) in map.iter().zip(remap) {
            for (node, renode) in row.iter().zip(rerow) {
                match (node, renode) {
                    (Some(want), Some(got)) => assert!(
                        ordered(*want).abs_diff(ordered(*got)) <= 1,
                        "node {want} read back as {got}"
                    ),
                    (None, None) => {}
                    _ => panic!("node availability changed"),
                }
            }
        }
    }
    assert_eq!(
        reparsed.to_ionex_string().expect("a written product rewrites"),
        encoded
    );
}

/// An `f64` mapped to a sign-magnitude-ordered `i64`, so adjacent floats differ
/// by one.
fn ordered(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        i64::MIN - bits
    } else {
        bits
    }
}

/// Reading `encoded` gives `original` without its skipped records, at the
/// exponent the writer chose, and writing that product again gives `encoded`.
fn assert_rereads_as(original: &Ionex, encoded: &str) {
    let reparsed = Ionex::parse(encoded.as_bytes()).expect("encoded IONEX must reparse");
    let mut expected = original.tec_grid_samples();
    expected.exponent = reparsed.exponent();
    // A product built from samples states no `# OF MAPS IN FILE`; the file the
    // writer wrote states one, so the product read back from it carries that
    // record.
    expected.header.maps_in_file = reparsed.header().maps_in_file;
    let expected = Ionex::from_samples(expected).expect("a product rebuilds from its samples");
    assert_eq!(reparsed, expected);
    assert_eq!(
        reparsed.to_ionex_string().expect("a written product rewrites"),
        encoded
    );
}

fn record(text: &mut String, data: &str, label: &str) {
    text.push_str(&format!("{data:<60}{label}\n"));
}
