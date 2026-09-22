#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::ephemeris::{Sp3, Sp3WriteError};

/// Whether a refusal is one an SP3 *file* can provoke.
///
/// `Sp3::parse` is deliberately more permissive than the canonical layout in a
/// few named places, so a product read from a file can hold something that text
/// cannot state back. Each arm below names the input that reaches it. Every
/// other refusal would be a writer regression on readable input - above all the
/// record-value refusals, which a parsed product cannot reach: the parser
/// requires each record value to be restatable in its own `F14.6` column and
/// routes the format's sentinels to absent values, so the writer's own record
/// checks must always pass here.
///
/// The epoch refusals are deliberately absent, and must stay absent. The writer
/// states an epoch only when the record it would emit reads back as the instant
/// the product holds, and for a parsed epoch it does: the calendar fields it
/// recovers are the fields the parser was handed, so the conversion back is the
/// same call on the same arguments. That holds for a UTC-like `23:59:60` label
/// too, which `split_julian_date` folds into the following day - the writer
/// offers the label alongside the ordinary next-day statement and keeps
/// whichever restates the stored split. An epoch refusal here means an input
/// this parser accepted and this writer cannot state back, which is exactly the
/// regression this target exists to find.
///
/// `YearNotRepresentable` and `DuplicateSatellite` are absent for the same
/// reason, and must stay absent. `Sp3::parse` bounds every epoch year to
/// `0..=9999` through `civil_datetime_with_second_policy`, the same range the
/// writer's own four-column check states, so no file states a year out of
/// range; the only way that refusal reaches this target is a day-number
/// derivation bug in the writer's own epoch candidates. `parse_plus_line`
/// keeps a satellite out of the list when it is already there, so no parsed
/// header carries a duplicate.
fn is_readable_input_refusal(error: &Sp3WriteError) -> bool {
    match error {
        // A header with no epoch records at all.
        Sp3WriteError::NoEpochs => true,
        // Line 2's seconds-of-week and epoch interval are kept even when the
        // field states an infinity or a NaN, so `validate_exact_sp3` can report
        // them as typed integrity failures.
        Sp3WriteError::NonFinite { .. } => true,
        // The MJD fraction is read without a column bound, and the two `%f`
        // bases are read at whatever precision their fields state.
        Sp3WriteError::NumberTooWide { .. } | Sp3WriteError::PrecisionNotRepresentable { .. } => {
            true
        }
        // The coordinate system, orbit type, agency, and comment text are read
        // from spans wider than the columns the layout gives them.
        Sp3WriteError::TextTooWide { .. } => true,
        // A label or comment carrying a control byte or a non-ASCII byte. The
        // parser reads UTF-8 and keeps the bytes between the columns.
        Sp3WriteError::TextNotColumnSafe { .. } => true,
        // A `P` header with `V` records under it: the parser keeps the velocity
        // it read, and a position product has no record to write it back in.
        Sp3WriteError::VelocityStateInPositionProduct { .. } => true,
        // Two `P` records for one satellite at one epoch, one an orbit and one
        // the missing-orbit sentinel with a clock. Retaining both is a separate
        // audit item; one record cannot state both.
        Sp3WriteError::ConflictingRecords { .. } => true,
        _ => false,
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(original) = Sp3::parse(data) else {
        return;
    };

    // A product read from a file writes back, unless the file stated something
    // in one of the places the parser is deliberately looser than the canonical
    // layout. Any other refusal is a writer regression, so it fails here rather
    // than being skipped.
    let encoded = match original.to_sp3_string() {
        Ok(encoded) => encoded,
        Err(error) => {
            assert!(
                is_readable_input_refusal(&error),
                "writer refused a product Sp3::parse produced: {error:?}"
            );
            return;
        }
    };
    let reparsed = Sp3::parse(encoded.as_bytes()).expect("encoded SP3 must reparse");

    // Serialization is idempotent: a product read back from text this writer
    // already accepted holds only values it accepts, so the re-encode succeeds
    // and matches byte for byte. Both the orbit records and the clock-only and
    // clock-rate records are part of that text, so a value dropped from either
    // shows up as a byte difference here and as a states/clock comparison
    // failure below.
    let re_encoded = reparsed
        .to_sp3_string()
        .expect("a product parsed from written SP3 must re-encode");
    assert_eq!(re_encoded, encoded);

    // `skipped_records` counts entries the input text carried but the product
    // cannot represent - an extended GLONASS slot such as `R28` beyond the
    // engine's PRN cap. Those are deliberately dropped instead of aborting the
    // parse (see `Sp3::skipped_records`), and nothing of them survives into the
    // product, so serialization has nothing to re-emit and a faithful re-encode
    // always reports zero. Asserting zero is stricter than comparing the two
    // counts: the writer must never emit a record that re-parses as
    // unrepresentable.
    assert_eq!(reparsed.skipped_records, 0);

    // Structural equality is asserted from the canonical generation onward, not
    // against `original`.
    //
    // `to_sp3_string` is a normalizing writer, not a verbatim echo: it emits the
    // standard header block (fixed `+`/`++`/`%c`/`%f`/`%i`/comment line counts)
    // and gives every header satellite a record at every epoch, using the
    // missing-orbit sentinel where the product holds no state. `Sp3` also
    // retains raw acquisition-validation provenance describing the *input text* -
    // `declared_satellite_tokens`, `epoch_position_tokens`,
    // `epoch_state_record_sequence`, the mandatory header-record counts and
    // `terminal_record` - plus `declared_num_epochs`. For a malformed or sparse
    // input those fields describe the original bytes, while the same fields on a
    // product parsed back from the normalized text describe the normalized
    // bytes, so `parse(write(x)) == x` is false by construction there and says
    // nothing about writer correctness. A conformant product is already
    // canonical and satisfies both forms.
    //
    // What must survive normalization is the product's meaning, so the
    // comparison is made over the public content - header, epoch instants,
    // comments, and every epoch's satellite states - rather than over the whole
    // struct. This still compares the original product against the one read
    // back from the writer's output, so a dropped satellite, a mangled
    // position, a lost epoch, or a clock that fails to re-read is caught.
    //
    // Reaching here means every header value was expressible, so the header
    // compares equal to itself and the comparison is meaningful.
    assert_eq!(reparsed.header, original.header);
    assert_eq!(reparsed.epochs, original.epochs);
    assert_eq!(reparsed.comments, original.comments);
    assert_eq!(reparsed.epoch_count(), original.epoch_count());
    for idx in 0..original.epoch_count() {
        assert_eq!(
            reparsed.states_at(idx).ok(),
            original.states_at(idx).ok(),
            "states differ at epoch {idx}"
        );
        // A satellite with no orbit but a valid clock is a record of its own,
        // and it is retained rather than folded into a zero position. It has to
        // come back too - with the microseconds the file stated, not a value
        // rounded on the way out.
        assert_eq!(
            reparsed.clock_records_at(idx).ok(),
            original.clock_records_at(idx).ok(),
            "clock-only records differ at epoch {idx}"
        );
    }
});
