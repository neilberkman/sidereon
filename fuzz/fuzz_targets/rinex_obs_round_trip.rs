#![no_main]

use std::collections::BTreeSet;

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::observations::{RinexObs, CYCLE_SLIP_FLAG};
use sidereon_core::GnssSystem;

// Round-trip class: a parsed observation product must re-encode to text that
// reparses to an equal product. A mismatch means the serializer is lossy or the
// parser accepts state the serializer cannot reproduce.
/// What a product holds that its writer cannot put back, judged from the
/// product rather than from the writer's refusal. At any version: an epoch
/// flag above 9, which no one-digit flag field holds. From version 3 to 4.01:
/// epoch picoseconds, which those epoch records have no field for. At version 2
/// also:
/// whether a downgrade removes it (scale factors, picoseconds, a clock offset
/// finer than `F12.9`), and whether nothing can (a year outside 1980 to 2079,
/// a clock offset too wide for `F12.9`). Returns (removable, permanent).
fn write_obstacles(product: &RinexObs) -> (bool, bool) {
    // Only an event may leave its epoch fields blank, at any version.
    let wide_flag = product.epochs().iter().any(|epoch| {
        epoch.flag > 9 || (epoch.epoch.is_none() && (epoch.flag <= 1 || epoch.flag == 6))
    });
    let version = product.header().version;
    // An event's header records that do not read leave every later list and
    // scale factor unknown; a product read from text never holds them.
    let timeline = product.header_timeline().ok();
    let unreadable = timeline.is_none();
    if version >= 3.0 {
        // Epoch records before 4.02 have no picosecond field and nothing
        // removes them; 4.02 records carry them after the clock.
        let picoseconds = version < 4.02
            && product
                .epochs()
                .iter()
                .any(|epoch| epoch.epoch_picoseconds.is_some());
        return (false, wide_flag || picoseconds || unreadable);
    }
    let clock_fits = |offset: f64| format!("{offset:12.9}").len() <= 12;
    let clock_exact = |offset: f64| format!("{offset:.9}").parse::<f64>() == Ok(offset);
    // A scale factor an event declares is refused at version 2 as the file
    // header's is.
    let event_scale_factors = timeline.as_ref().is_some_and(|timeline| {
        timeline
            .segments()
            .any(|(_, header)| !header.scale_factors.is_empty())
    });
    let removable = event_scale_factors
        || !product.header().scale_factors.is_empty()
        || holds_unstated_list(product)
        || product.epochs().iter().any(|epoch| {
            epoch.epoch_picoseconds.is_some()
                || epoch
                    .rcv_clock_offset_s
                    .is_some_and(|offset| !clock_exact(offset))
        });
    let permanent = wide_flag
        || unreadable
        || product.epochs().iter().any(|epoch| {
            epoch
                .epoch
                .is_some_and(|time| !(1980..=2079).contains(&time.year))
                || epoch
                    .rcv_clock_offset_s
                    .is_some_and(|offset| !offset.is_finite() || !clock_fits(offset))
        });
    (removable, permanent)
}

/// Whether a version 2 product holds a code list its file may not state: one
/// for a constellation no observation or count names, other than the one a
/// file with no observations names in its version record, which is the held
/// constellation, else the product's one list's, else GPS. The writer refuses
/// it, and a downgrade removes it, only when the type names it writes do not
/// read as the list, which this does not decide.
fn holds_unstated_list(product: &RinexObs) -> bool {
    let header = product.header();
    // A cycle slip record names its satellite on the epoch line as an
    // observation record does.
    let mut stated: BTreeSet<GnssSystem> = product
        .epochs()
        .iter()
        .flat_map(|epoch| {
            epoch
                .sats
                .keys()
                .chain(epoch.cycle_slips.keys())
                .map(|sat| sat.system)
        })
        .collect();
    if stated.is_empty() {
        let mut lists = header.obs_codes.keys();
        let fallback = header
            .rinex2_system
            .unwrap_or(match (lists.next(), lists.next()) {
                (Some(only), None) => *only,
                _ => GnssSystem::Gps,
            });
        stated.insert(fallback);
    }
    stated.extend(
        header
            .prn_obs_counts
            .iter()
            .filter(|(_, counts)| !counts.is_empty())
            .map(|(sat, _)| sat.system),
    );
    header
        .obs_codes
        .keys()
        .any(|system| !stated.contains(system))
}

/// Header records an event may carry that take effect for later epochs, at
/// either version, and the labels a reader recognises out of their columns.
const EVENT_LABELS: &[&str] = &[
    "SYS / # / OBS TYPES",
    "# / TYPES OF OBSERV",
    "SYS / SCALE FACTOR",
    "SYS / PHASE SHIFT",
    "GLONASS SLOT / FRQ #",
    "GLONASS COD/PHS/BIS",
    "INTERVAL",
    "MARKER NAME",
    "MARKER NUMBER",
    "MARKER TYPE",
    "APPROX POSITION XYZ",
    "ANTENNA: DELTA H/E/N",
    "ANT # / TYPE",
    "REC # / TYPE / VERS",
    "OBSERVER / AGENCY",
    "SIGNAL STRENGTH UNIT",
    "COMMENT",
];

/// A record's label in its columns, or `None` when the record is not
/// printable ASCII or names a label out of its columns, which a reader moves
/// into them and this check does not.
fn record_label(record: &str) -> Option<&str> {
    if !record
        .bytes()
        .all(|byte| byte == b' ' || byte.is_ascii_graphic())
    {
        return None;
    }
    let label = record.get(60..).map_or("", str::trim);
    let trimmed = record.trim_end();
    let misplaced = EVENT_LABELS
        .iter()
        .any(|known| trimmed.ends_with(known) && label != *known);
    (!misplaced).then_some(label)
}

fn takes_effect(version: f64, label: &str) -> bool {
    match label {
        "SYS / # / OBS TYPES" => version >= 3.0,
        "# / TYPES OF OBSERV" => version < 3.0,
        "COMMENT" | "" => false,
        _ => EVENT_LABELS.contains(&label),
    }
}

fn continues_declaration(label: &str, previous: &str, record: &str) -> bool {
    if label != previous {
        return false;
    }
    let blank = |end: usize| record.get(..end).is_none_or(|head| head.trim().is_empty());
    match label {
        "SYS / # / OBS TYPES" | "SYS / SCALE FACTOR" => blank(1),
        "# / TYPES OF OBSERV" => blank(6),
        "SYS / PHASE SHIFT" => blank(18),
        "GLONASS SLOT / FRQ #" => blank(3),
        _ => false,
    }
}

/// Labels whose declarations give values to some satellites, codes, slots or
/// bias codes each.
const KEYED_LABELS: &[&str] = &[
    "SYS / PHASE SHIFT",
    "SYS / SCALE FACTOR",
    "GLONASS SLOT / FRQ #",
    "GLONASS COD/PHS/BIS",
];

/// Labels whose records each set one value the product applies, which a header
/// block may not give two values.
const SINGLE_VALUE_LABELS: &[&str] = &[
    "INTERVAL",
    "MARKER NAME",
    "MARKER NUMBER",
    "MARKER TYPE",
    "APPROX POSITION XYZ",
    "ANTENNA: DELTA H/E/N",
    "ANT # / TYPE",
    "REC # / TYPE / VERS",
    "OBSERVER / AGENCY",
    "SIGNAL STRENGTH UNIT",
    "RINEX VERSION / TYPE",
    "TIME OF FIRST OBS",
    "TIME OF LAST OBS",
    "LEAP SECONDS",
    "# OF SATELLITES",
];

/// What a declaration gives a value to, judged from its first record: its label
/// with, for a type list, scale factor or phase shift, its constellation. Two declarations with one key may give one
/// satellite, code, slot or bias code a value, and this looks no further, so it
/// takes them to. A comment, or a record whose value nothing applies, gives
/// nothing a value.
fn declaration_key(record: &str) -> Option<(String, String)> {
    let label = record_label(record)?;
    let content = record.get(..60).unwrap_or(record);
    let scope = match label {
        // A phase shift record naming only its constellation covers every code
        // of it, so every phase shift of a constellation shares a key.
        "SYS / # / OBS TYPES" | "SYS / SCALE FACTOR" | "SYS / PHASE SHIFT" => {
            content.trim_start().chars().take(1).collect()
        }
        "# / TYPES OF OBSERV" | "GLONASS SLOT / FRQ #" | "GLONASS COD/PHS/BIS" => String::new(),
        single if SINGLE_VALUE_LABELS.contains(&single) => String::new(),
        _ => return None,
    };
    Some((label.to_string(), scope))
}

/// An event's records as declarations, each record with its continuations, or
/// `None` when a record's label is not in its columns.
fn declarations(records: &[String]) -> Option<Vec<Vec<String>>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut previous = "";
    for record in records {
        let label = record_label(record)?;
        match groups.last_mut() {
            Some(group) if continues_declaration(label, previous, record) => {
                group.push(record.clone());
            }
            _ => groups.push(vec![record.clone()]),
        }
        previous = label;
    }
    Some(groups)
}

/// Header records giving values to different keys give the header in effect
/// whether they share one event or are split across consecutive events,
/// declaration by declaration, and the file written from each reads the same
/// union, headers and values. Declarations that may share a key are not split:
/// within one event they have no order, and across events the later one
/// replaces the earlier.
fn check_split_events(original: &RinexObs, encoded: Option<&str>) {
    let Ok(timeline) = original.header_timeline() else {
        return;
    };
    let read_original = encoded.map(|text| {
        let read = RinexObs::parse(text).expect("encoded RINEX OBS must reparse");
        let timeline = read
            .header_timeline()
            .expect("a written file's headers read");
        (read, timeline)
    });
    for (index, epoch) in original.epochs().iter().enumerate() {
        if !(2..=5).contains(&epoch.flag) {
            continue;
        }
        let Some(groups) = declarations(&epoch.special_records) else {
            continue;
        };
        if groups.len() < 2 {
            continue;
        }
        let keys: Vec<_> = groups
            .iter()
            .filter_map(|group| declaration_key(&group[0]))
            .collect();
        if keys.iter().collect::<BTreeSet<_>>().len() != keys.len() {
            continue;
        }
        let extra = groups.len() - 1;
        let copies: Vec<_> = groups
            .into_iter()
            .map(|group| {
                let mut copy = epoch.clone();
                copy.declared_record_count = group.len();
                copy.special_records = group;
                copy
            })
            .collect();
        let mut split = original.clone();
        split.epochs.splice(index..=index, copies);
        let split_timeline = split
            .header_timeline()
            .expect("an event's declarations read in consecutive events");
        for later in index..original.epochs().len() {
            assert_eq!(
                split_timeline.at(later + extra),
                timeline.at(later),
                "the header at epoch {later} after splitting event {index}"
            );
        }
        let Some((read_original, read_original_timeline)) = &read_original else {
            continue;
        };
        let split_text = split
            .to_rinex_string()
            .expect("an event's declarations write in consecutive events");
        let read_split = RinexObs::parse(&split_text).expect("the split file must reparse");
        assert_eq!(
            read_split.header().obs_codes,
            read_original.header().obs_codes,
            "the union after splitting event {index}"
        );
        let read_split_timeline = read_split
            .header_timeline()
            .expect("a written file's headers read");
        for later in index..original.epochs().len() {
            assert_eq!(
                read_split_timeline.at(later + extra),
                read_original_timeline.at(later),
                "the header read at epoch {later} after splitting event {index}"
            );
            assert_eq!(
                values_by_code(&read_split, later + extra),
                values_by_code(read_original, later),
                "values read at epoch {later} after splitting event {index}"
            );
        }
    }
}

/// One epoch's values and slips by satellite and code copy.
fn values_by_code(product: &RinexObs, index: usize) -> Vec<(String, String, String)> {
    let mut held = Vec::new();
    let epoch = &product.epochs()[index];
    for (kind, records) in [("obs", &epoch.sats), ("slip", &epoch.cycle_slips)] {
        for (sat, values) in records {
            let Some(codes) = product.header().obs_codes.get(&sat.system) else {
                continue;
            };
            let mut seen = std::collections::BTreeMap::<&str, usize>::new();
            for (code, value) in codes.iter().zip(values) {
                let copy = seen.entry(code.as_str()).or_default();
                *copy += 1;
                if value.value.is_some() || value.lli.is_some() || value.ssi.is_some() {
                    held.push((
                        format!("{kind} {sat}"),
                        format!("{code}#{copy}"),
                        format!("{value:?}"),
                    ));
                }
            }
        }
    }
    held.sort();
    held
}

/// The header records of an event before the first epoch give the header those
/// records give moved into the file header, wherever the file header takes
/// them. A file header declaration with a key the event's records give a value,
/// a type list for the same constellation or a record of a label holding one
/// value, is taken out of the header, as the event replaces it. Where a file
/// header phase shift, scale factor, GLONASS slot or GLONASS bias record may
/// share a key with the event's, the move is not checked: in one header block
/// the two would apply by their keys or contradict each other.
fn check_leading_event_moves(original: &RinexObs, encoded: &str) {
    let Some(first) = original.epochs().first() else {
        return;
    };
    if !(2..=5).contains(&first.flag) {
        return;
    }
    let version = original.header().version;
    let mut moved_records = Vec::new();
    for record in &first.special_records {
        let Some(label) = record_label(record) else {
            return;
        };
        if takes_effect(version, label) {
            moved_records.push(record.clone());
        }
    }
    let lines: Vec<&str> = encoded.lines().collect();
    let Some(end) = lines
        .iter()
        .position(|line| line.ends_with("END OF HEADER"))
    else {
        return;
    };
    let Some(event_declarations) = declarations(&moved_records) else {
        return;
    };
    let event_keys: BTreeSet<(String, String)> = event_declarations
        .iter()
        .filter_map(|group| declaration_key(&group[0]))
        .collect();
    let header_lines: Vec<String> = lines[..end]
        .iter()
        .map(|line| (*line).to_string())
        .collect();
    let Some(header_declarations) = declarations(&header_lines) else {
        return;
    };
    let mut text: Vec<String> = Vec::new();
    for group in header_declarations {
        match declaration_key(&group[0]) {
            Some(key) if event_keys.contains(&key) => {
                if KEYED_LABELS.contains(&key.0.as_str()) {
                    return;
                }
            }
            _ => text.extend(group),
        }
    }
    text.extend(moved_records);
    text.push(lines[end].to_string());
    let body = end + 2 + first.special_records.len();
    let Some(rest) = lines.get(body..) else {
        return;
    };
    text.extend(rest.iter().map(|line| (*line).to_string()));
    // A record the file header's form does not take has nothing to compare.
    let Ok(moved) = RinexObs::parse(&text.join("\n")) else {
        return;
    };
    let Ok(at) = original.header_at(0) else {
        return;
    };
    let file = moved.header();
    assert_eq!(file.approx_position_m, at.approx_position_m);
    assert_eq!(file.antenna_delta_hen_m, at.antenna_delta_hen_m);
    assert_eq!(file.declared_obs_codes, at.declared_obs_codes);
    assert_eq!(file.rinex2_types, at.rinex2_types);
    assert_eq!(file.marker_name, at.marker_name);
    assert_eq!(file.marker_number, at.marker_number);
    assert_eq!(file.marker_type, at.marker_type);
    assert_eq!(file.observer, at.observer);
    assert_eq!(file.agency, at.agency);
    assert_eq!(file.receiver, at.receiver);
    assert_eq!(file.antenna, at.antenna);
    assert_eq!(
        file.interval_s.map(f64::to_bits),
        at.interval_s.map(f64::to_bits)
    );
    assert_eq!(file.phase_shifts, at.phase_shifts);
    assert_eq!(file.scale_factors, at.scale_factors);
    assert_eq!(file.glonass_slots, at.glonass_slots);
    assert_eq!(file.glonass_cod_phs_bis, at.glonass_cod_phs_bis);
    assert_eq!(file.signal_strength_unit, at.signal_strength_unit);
    assert_eq!(moved.epochs().len() + 1, original.epochs().len());
    for index in 1..original.epochs().len() {
        assert_eq!(
            values_by_code(&moved, index - 1),
            values_by_code(original, index),
            "values at epoch {index} with the leading event in the header"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = RinexObs::parse(&text) else {
        return;
    };
    // Records skipped as unrepresentable are not re-emitted, so they would not
    // survive a round trip; restrict the invariant to clean products.
    if original.skipped_records != 0 {
        return;
    }
    // A product may hold what its writer cannot put back: the reader keeps a
    // flag wider than the flag field, and takes version 3 epoch records and
    // scale factors in a version 2 file. A refusal is accepted only when the
    // product shows why; at version 2 the downgrade then writes it when what
    // stands in the way is removable and refuses it when it is not, and a
    // downgrade that writes it has to report a change, since one of a product
    // the writer takes reports none. Any other refusal fails.
    let encoded = match original.to_rinex_string() {
        Ok(encoded) => {
            assert!(
                !write_obstacles(&original).1,
                "wrote a product holding what its writer cannot put back"
            );
            // The header in effect cannot depend on how an event's records
            // are grouped or where a leading event's records are.
            check_split_events(&original, Some(&encoded));
            check_leading_event_moves(&original, &encoded);
            encoded
        }
        Err(error) if original.header().version < 3.0 => {
            check_split_events(&original, None);
            let (removable, permanent) = write_obstacles(&original);
            assert!(
                removable || permanent,
                "refused a version 2 product holding nothing version 2 cannot: {error}"
            );
            let version = original.header().version;
            match original.downgrade_to_rinex2(version) {
                Ok((downgraded, changes)) => {
                    assert!(!permanent, "downgraded what version 2 cannot hold");
                    assert!(
                        !changes.is_empty(),
                        "refused a version 2 product its downgrade writes unchanged: {error}"
                    );
                    let text = downgraded
                        .to_rinex_string()
                        .expect("serialize the downgrade");
                    RinexObs::parse(&text).expect("the written downgrade must reparse");
                }
                Err(refusal) => assert!(permanent, "the downgrade refused: {refusal}"),
            }
            return;
        }
        Err(error) => {
            check_split_events(&original, None);
            assert!(write_obstacles(&original).1, "serialize RINEX OBS: {error}");
            return;
        }
    };
    let mut reparsed = RinexObs::parse(&encoded).expect("encoded RINEX OBS must reparse");
    // The labels of header records that were read and not retained describe the
    // source text rather than the product, and the writer does not re-emit them,
    // so a clean re-parse reports none. Normalise that one diagnostic rather
    // than skipping the whole product, which would stop checking its records.
    let mut original = original;
    original.header.unretained_header_labels.clear();
    reparsed.header.unretained_header_labels.clear();
    // The count an epoch line declared describes the source text too: the
    // writer states the records it has, so two records for one satellite are
    // declared as the one kept.
    for epoch in original.epochs.iter_mut().chain(reparsed.epochs.iter_mut()) {
        epoch.declared_record_count = if epoch.flag == CYCLE_SLIP_FLAG {
            epoch.cycle_slips.len()
        } else if epoch.flag > 1 {
            epoch.special_records.len()
        } else {
            epoch.sats.len()
        };
    }
    // A version 2 version record names one constellation only while every
    // observation is from it; otherwise the writer names `M (MIXED)`, which
    // reads back as naming none.
    if let Some(system) = original.header.rinex2_system {
        if original.epochs.iter().any(|epoch| {
            epoch
                .sats
                .keys()
                .chain(epoch.cycle_slips.keys())
                .any(|sat| sat.system != system)
        }) {
            original.header.rinex2_system = None;
        }
    }
    assert_eq!(reparsed, original);
});
