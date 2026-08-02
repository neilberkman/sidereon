#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::ephemeris::Sp3;

fuzz_target!(|data: &[u8]| {
    let Ok(original) = Sp3::parse(data) else {
        return;
    };

    let encoded = original.to_sp3_string();
    let reparsed = Sp3::parse(encoded.as_bytes()).expect("encoded SP3 must reparse");

    // Serialization is idempotent for every product, including one carrying a
    // non-finite header value.
    let re_encoded = reparsed.to_sp3_string();
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
    // `Sp3` compares its f64 header fields, so a product whose seconds-of-week
    // or epoch interval is non-finite - which `Sp3::parse` deliberately keeps
    // for `validate_exact_sp3` to reject as a typed integrity failure - is not
    // equal to itself, and equality says nothing about it. The byte-idempotence
    // assertion above still covers those products.
    let comparable = original.header == original.header.clone();
    if comparable {
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
        }
    }
});
