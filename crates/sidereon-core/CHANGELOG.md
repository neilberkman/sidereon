# Changelog

All notable changes to `sidereon-core` are documented here.

## [Unreleased]

### Changed

- **Breaking.** `Tdm` and `TdmMetadata` hold `comments: Vec<TdmComment>` rather
  than `Vec<String>`, preserving the position of header and metadata comments
  among fields. Comments are written back in their original position rather than
  gathered to the top of their block. A header or metadata comment positioned
  away from the start of its block (CCSDS 503.0-B-2 4.5.2 a), 4.5.2 b)) is
  refused under strict policy with `TdmError::KeywordOutOfOrder` and emitted as
  `TdmDeparture::KeywordOutOfOrder` when `keyword_order` is forgiven.
- A comment's text retains leading whitespace (indentation) following the
  required single space after `COMMENT` (CCSDS 503.0-B-2 4.5.3). Trailing
  whitespace at the end of a comment line remains insignificant (4.2.9). A bare
  `COMMENT` line represents an empty comment and encodes as a bare `COMMENT`
  line.
- **Breaking.** A timetag carrying the `Z` time code terminator alongside a
  `TIME_SYSTEM` other than `UTC` is refused under every policy with
  `TdmError::MalformedEpoch`. CCSDS 503.0-B-2 4.3.9 permits `Z` only for UTC;
  declaring another time system while suffixing with Zulu specifies an instant
  it is not. None of the 53 files in the test corpus carry `Z` alongside a
  non-UTC time system.
- Timetag comparison and uniqueness keys for tracking data records (CCSDS
  503.0-B-2 3.4.10, 3.4.11) are built entirely from integer components: civil
  day, whole seconds of day, and fractional seconds as an aligned integer.
  Timetags differing in the last decimal place of fractional seconds no longer
  compare equal or alias into duplicate records. Leap second timetags
  (`23:59:60`) preserve second-of-minute as written without cross-day
  normalisation into the next day's `00:00:00`.
- **Breaking.** `tdm::encode_kvn` and `tdm::encode_kvn_with_policy` enforce the
  reader's rules over what they are about to write, refusing a value by name
  under `TdmWritePolicy::strict()` and reporting a named departure under a
  policy that forgives that axis. The writer previously checked only line caps
  and character sets, so a caller-built message carrying structural errors,
  records out of chronological order (3.4.10), duplicate records (3.4.11),
  keywords out of table order (3.2.3 and 3.3.1.8), missing mandatory keywords
  (3.1.3, 3.3.1.7, Table 3-2, Table 3-3), a `PATH` naming an undefined
  participant (3.3.1.9), indexed keywords outside Table 3-3 range, an empty
  mandatory value (4.3.1), or a malformed timetag (4.3.9) was written out into
  a non-conforming file. A data-section comment after the first record
  (4.5.2 c)) is now refused under strict policy with
  `TdmError::KeywordOutOfOrder` and emitted as `TdmDeparture::KeywordOutOfOrder`
  when `keyword_order` is forgiven. `TdmWritePolicy::as_read` translates the
  write policy to the reader's vocabulary for mirrored validation.
- **Breaking.** `tdm::encode_kvn` and `tdm::parse_kvn` refuse a keyword repeated
  in one header or metadata block with different values using the new
  `TdmError::ConflictingKeyword`. CCSDS 503.0-B-2 4.2.5 a) gives each keyword "a
  single value assignment", and 3.2.3 and 3.3.1.8 give it one place in its
  table's order; choosing between two values that disagree would be inventing
  data, which no policy forgives.
- `TdmWritePolicy` gained `repeated_keywords: TdmLeniency`. A keyword written
  twice in one block with the same value violates CCSDS 503.0-B-2 4.2.5 a); the
  writer refuses it under `TdmWritePolicy::strict()` with the new
  `TdmError::RepeatedKeyword` and emits `TdmDeparture::RepeatedKeyword` when
  forgiven. The reader has no policy axis for repeated keywords because no file
  in the 53-file corpus repeats one; it takes identical repeats unconditionally
  and reports `TdmWarning::RepeatedKeyword`.
- **Breaking.** `TdmError::MalformedEpoch`, `TdmError::KeywordOutOfOrder`, and
  `TdmError::EmptyValue` carry `line: Option<usize>` in place of `line: usize`.
  The writer raises each with no input line to point at when validating
  caller-built messages. A read fills `line` with `Some` and the input line number.
- `tdm::encode_kvn_with_policy` writes a TDM under a `TdmWritePolicy`,
  returning the text together with the departures from CCSDS 503.0-B-2 it
  emitted as `TdmDeparture` values. The default emits none, so `tdm::encode_kvn`
  is unchanged and refuses a value it cannot write conformingly. A field set to
  `TdmLeniency::Forgive` lets the writer emit one departure and names it, so a
  caller asking for a non-conforming file is told exactly what makes it one. The
  set mirrors what `TdmPolicy` forgives on the way in, in the same vocabulary,
  so a message read leniently can be written back by asking for the same
  departures. Nothing outside that mirror is emittable under any policy: a field
  keyed `COMMENT`, a key holding an equals sign or whitespace, a comment
  carrying a newline, a value that does not parse as its keyword's type. Those
  produce a file that reads back as something other than what was written, which
  no policy can make correct.
- **Breaking.** `tdm::encode_kvn` refuses a line it would write that departs
  from CCSDS 503.0-B-2 4.2.1: one holding a character outside printable ASCII,
  or one over 254 characters. Both reach the writer from a message read under a
  lenient `non_printable` or `long_lines`, since a character inside a comment or
  a free-text value is part of the value and a long value keeps its length, and
  neither can be shortened or dropped without losing what the value says. Under
  a matching `TdmWritePolicy` the writer emits the line and names it as
  `TdmDeparture::NonPrintableCharacter` or `TdmDeparture::LineTooLong`. Both
  departures existed and neither was ever produced, so a message carrying either
  was written non-conforming and unreported.
- **Breaking.** `TdmError::NonPrintableCharacter` and `TdmError::LineTooLong`
  carry `line: Option<usize>` in place of `line: usize`, and each gained
  `keyword: String`. The writer raises both with no input line to point at, and
  the keyword is the locator it does have: the first token of the line it built.
  A read fills `line` with `Some` and the keyword with the offending line's
  first token.
- **Breaking.** A TDM whose last line carries no terminator is refused with the
  new `TdmError::UnterminatedFinalLine`, and forgiven under a lenient
  `final_terminator`. CCSDS 503.0-B-2 4.2.11 terminates every line, the last one
  included; 13 of the 53 public files gathered for this audit end without one,
  10 of those indent with tabs as well, and seven read once both are forgiven.
  None of the 53 turns on the terminator alone, since the other six are refused
  for what they say rather than for how they end, so the axis earns its place as
  the second permission those seven need and on the class generally.
  `tdm::encode_kvn` terminates its last line, which it did not: every message
  the writer produced ended unterminated, so sidereon
  emitted the departure it now names. The missing terminator is reported behind
  every failure about what the message says: an unclosed block, an absent
  `CCSDS_TDM_VERS`, `CREATION_DATE` or `ORIGINATOR`, a version outside the `x.y`
  form, and a message with no segment. A character outside printable ASCII and a
  line over 254 characters stay ahead of those, since each names a position in
  one line and explains how the rest of that line reads.
- **Breaking.** `tdm::encode_kvn` refuses a field or comment the KVN form
  cannot carry, with the new `TdmError::Unwritable` naming the keyword and the
  reason. `TdmField` and the comment vectors are public, so a caller can hold a
  key that is empty, holds an equals sign, or is padded with whitespace, a value
  padded the same way, or either carrying a line terminator; each was written
  out and read back as a different value, which is the defect this work exists
  to close. The writer also checks keyword membership per section as the reader
  does: the mandatory-keyword check looked for a `PARTICIPANT_` prefix, so a
  caller-built `PARTICIPANT_9` satisfied it and was written out.
- **Breaking.** A TDM data-section comment keeps its place. `TdmDataSection`
  holds `Vec<TdmComment>` rather than `Vec<String>`, each carrying the index of
  the record it precedes, and the writer puts each one back where it was read.
  Comments were gathered to the top of the block on write, so a caller who read
  a file, changed one record and wrote it back got a rearranged file; nothing
  reported the move. CCSDS 503.0-B-2 4.5.2 c) puts a data-section comment
  between `DATA_START` and the first record, so one after a record is refused
  with `TdmError::KeywordOutOfOrder` naming the line, and forgiven under a
  lenient `keyword_order` with the same warning. Keeping the position is what
  makes the forgiven message write back to the bytes it came from.
- **Breaking.** TDM keywords out of the order tables 3-2 and 3-3 fix are
  refused with the new `TdmError::KeywordOutOfOrder`. 3.2.3 and 3.3.1.8 make
  that order binding and nothing checked it, so a header comment written after
  `ORIGINATOR`, where 4.5.2 a) puts it before `CREATION_DATE`, read as if it
  were in place. Order changes no value, so it is forgivable: a lenient read
  takes the message and reports each keyword out of place by line. An indexed
  keyword ranks where its base does, so `PARTICIPANT_2` beside `PARTICIPANT_1`
  is in order whichever comes first. Seven of the 32 producer files among the 53
  gathered for this audit deviate, three in the header and six in metadata; none
  of the 21 annex examples does, so this is a place the standard and practice
  have parted company rather than one where the standard contradicts itself.
- **Breaking.** A TDM `PATH` naming a participant index its segment does not
  define is refused with the new `TdmError::UndefinedParticipant`. A path entry
  is a participant number, and one that resolves to nothing left the
  measurements it describes belonging to no participant, with the reader
  carrying the path on regardless. It is refused under every policy, lenient
  included, since choosing which participant was meant would be inventing one.
  A gap in the indices is untouched: CCSDS 503.0-B-2 3.3.1.9 requires them to
  differ, not to run consecutively, so `PARTICIPANT_1` beside `PARTICIPANT_3`
  with nothing pointing into the gap stays legal. No public file on hand carries
  a dangling reference.
- **Breaking.** `TdmError` and `TdmWarning` carry one payload convention. A
  keyword is `keyword: String` in every variant, whether this crate named it or
  read it from the message; it was `key`, `field` or `keyword` by turns, and
  `&'static str` in the variants this crate named and `String` in the ones it
  read, so a caller rendering an error had to know which. A label this crate
  classifies with rather than reads, `section` and `detail`, stays
  `&'static str`. Each variant names where the problem is with the most specific
  locator meaningful in both directions: `line` where only the reader can raise
  it, `line: Option<usize>` where the writer raises the same failure with no
  input line, and `segment` where the problem belongs to a segment. The
  `#[non_exhaustive]` annotation makes a later variant additive but does not
  make changing an existing payload free, which is why this lands with the rest
  of the breaking work rather than after it.
- **Breaking.** An absent `CCSDS_TDM_VERS` reports `TdmError::MissingKeyword`
  like every other mandatory keyword, and `TdmError::MissingVersion` is gone. It
  is one of the keywords table 3-2 marks mandatory, and a caller asking which
  mandatory keyword a message lacks should not need a second match arm for the
  one that happens to come first. `TdmError::InvalidVersion` carries the line it
  was read from, as every other parse error does, or `None` for a value a caller
  built that no input produced. The version was also filtered for emptiness
  before being reported absent, which became unreachable once an empty value was
  refused at the line that gives it.
- **Breaking.** TDM tracking data records are read by the timetags CCSDS
  503.0-B-2 4.3.9 defines, and ordered and deduplicated by them. The epoch was
  kept as a raw string and never checked, so any text passed and no record could
  be placed in time. Both forms, `YYYY-MM-DDThh:mm:ss[.d->d][Z]` and
  `YYYY-DDDThh:mm:ss[.d->d][Z]`, are read and compare against each other, so a
  message may write either; a timetag outside them is the new
  `TdmError::MalformedEpoch`. 3.4.10 requires each keyword's records to be in
  chronological order and 3.4.11 requires each keyword and timetag pair to be
  unique; neither was checked, and a repeat passed through with both values kept
  and nothing saying so. They are now `TdmError::RecordsOutOfOrder` and
  `TdmError::DuplicateRecord`, both forgivable: a data section is a sequence
  rather than a set of keyed slots, so a lenient read keeps both records in file
  order and warns, inventing nothing. Figure E-17 needs exactly that, giving
  `RCS` twice at `2011-05-11T10:26:33.7008` with different values. The writer
  refuses a repeat whatever the reader forgave, so a forgiven message is never
  written back in a form the standard forbids. `TdmPolicy::strict` is a constant
  form of the default, for callers building a policy in a `const`.
- **Breaking.** A TDM keyword outside the table for its section is refused.
  3.2.3 says "Only those keywords shown in table 3-2 shall be used in a TDM
  Header" and 3.3.1.7 says the same of table 3-3 and a metadata section; both
  catch-alls accepted anything and wrote it back, so a misspelled keyword
  travelled through the reader carrying data whose meaning nothing defines. It
  is now the new `TdmError::UndefinedKeyword`, naming the line, the keyword and
  the section. This is refused under every policy, lenient included, since
  reading it would mean inventing what it means. `EPHEMERIS_NAME` is taken
  unindexed as well as indexed: table 3-3 lists only `EPHEMERIS_NAME_n`, but
  figure E-17 writes the bare keyword and the annex I summary sheet lists it
  bare five times. The four section markers join `COMMENT` as keywords 4.2.5 c)
  excepts from the KVN syntax, so `META_START = 1` and its three companions are
  refused as malformed lines on read and as `TdmError::KeywordNotAssignable` on
  write, where only `COMMENT` was caught before.
- **Breaking.** TDM indexed keywords are bounded by the ranges their tables
  give. Table 3-3 indexes `PARTICIPANT_n` with n = {1,2,3,4,5} and 3.3.1.11 caps
  a segment at five participants, and the table defines `PATH`, `PATH_1` and
  `PATH_2` and no other index; any suffix that fit a `u8` was accepted, so
  `PARTICIPANT_0`, `PARTICIPANT_99` and `PATH_03` all parsed and wrote back out.
  A suffix outside the range is refused with `TdmInputErrorKind::InvalidIndex`,
  as the data keywords already were. A padded suffix such as `PARTICIPANT_01` is
  refused on the same footing: the table writes the indexer as a single digit,
  and reading `01` as 1 gave one index two spellings that the writer then
  emitted unchanged. 3.3.1.9 says "The indexer shall not be the same for any two
  participants in a given Metadata Section"; both were kept and the second
  silently shadowed the first wherever an index is resolved. Two participants
  sharing an indexer is the same keyword written twice, so it is refused as
  `TdmError::ConflictingKeyword`, naming the line and both values; an identical
  repeat reports `TdmWarning::RepeatedKeyword` and yields one participant.
- `tdm::parse_kvn_with_policy` reads a TDM under a `TdmPolicy`, returning the
  message together with the departures from CCSDS 503.0-B-2 it forgave as
  `TdmWarning` values. The default policy forgives nothing, so `tdm::parse_kvn`
  is unchanged and returns no warnings. A field set to `TdmLeniency::Forgive`
  forgives a departure that does not change what a value means: a character
  outside printable ASCII, a keyword tables 3-2 and 3-3 mark mandatory that the
  message omits, a line over the 254 characters 4.2.1 allows, and a data section
  holding none of the records 3.1.3 requires. Nothing that
  changes what the message means is forgivable under any policy, including a
  value that does not parse, a unit contradicting table 3-5, an undefined
  keyword carrying data, a structural error, and a `COMMENT` used as an
  assignment key. The writer stays strict whatever the reader forgave: a
  leniently read message either writes back unchanged or is refused naming what
  the standard forbids, never quietly repaired. This follows the shape the IONEX
  reader uses for its own policies and warnings.
- **Breaking.** A TDM carries the records and keywords CCSDS 503.0-B-2 makes
  mandatory, or it is refused by name. Table 3-2 marks `CREATION_DATE` and
  `ORIGINATOR` mandatory, table 3-3 marks `TIME_SYSTEM` mandatory and
  `PARTICIPANT_n` mandatory with "at least one", and 3.1.3 gives each segment a
  data section of "a minimum of one Tracking Data Record"; none was required, so
  a message missing any of them parsed and wrote back out still missing it. The
  absent keyword is now `TdmError::MissingKeyword`, naming the keyword and the
  segment that wanted it, and an empty data block is
  `TdmError::EmptyDataSection`. 4.3.1 requires "A non-empty value field must be
  specified for each keyword provided"; `KEY =` was read as an empty value,
  which the modeled header fields then dropped on write so the message lost a
  keyword it had declared, and is now `TdmError::EmptyValue`. 3.2.5 gives the
  version "the form of x.y"; any text passed, and a value outside that form is
  now `TdmError::InvalidVersion`. `tdm::encode_kvn` applies the same rules to a
  value built by a caller, looking for the mandatory metadata keywords in the
  fields it writes rather than in the parsed properties beside them, so an
  encoding that would omit one is refused instead of produced.
- **Breaking.** A TDM is read in the lines and characters CCSDS 503.0-B-2
  defines. 4.2.11 terminates a line with "a single Carriage Return or a single
  Line Feed or a Carriage Return/Line Feed pair or a Line Feed/Carriage Return
  pair"; the reader split on line feeds alone, so a file written with carriage
  returns read as one long line and was refused for having no
  `CCSDS_TDM_VERS`. Each pair now ends one line rather than leaving an empty
  line behind it. 4.2.1 allows "only printable ASCII characters and blanks",
  says "ASCII control characters (such as TAB, etc.) must not be used", and caps
  a line at 254 characters excluding its terminator; none of that was checked,
  so a tab, a NUL or a typographic quotation mark travelled through the reader
  into a parsed value and back out of the writer. A line holding one is refused
  with the new `TdmError::NonPrintableCharacter`, naming the line, the column
  and the character, and an over-long line with the new `TdmError::LineTooLong`.
  Two public TDM corpora carry a right double quotation mark in a clock-offset
  comment, and those files are now refused by name rather than read.
- **Breaking.** `TdmError` and `TdmInputErrorKind` are `#[non_exhaustive]`, so a
  caller matching on either needs a wildcard arm. The CCSDS 503.0-B-2 audit adds
  failure modes as it covers more of the standard, and the annotation is what
  makes those later additions additive rather than a compatibility event each
  time. The refusal of a field whose key cannot carry a value is now its own
  variant, `TdmError::KeywordNotAssignable`, naming the key as the field holds
  it. It was reported as `TdmError::MalformedLine` carrying a line number
  counted from an encoding that is never produced, which said a line was
  malformed in a stream the caller could not look at; the distinction a caller
  needs is between a line that is not a KVN assignment and a key that the
  standard does not let carry a value at all. `tdm::encode_kvn` raises it from
  `validate_tdm`, with the whole value checked before any line is built.
- An IONEX value scales as the reference readers scale it, `field * 10^EXPONENT`,
  with the factor built from an exact power of ten rather than taken from a
  `pow` implementation, so a node does not depend on a library's rounding: `10^n`
  is exact in a double up to `n` of 22, and `1.0` over an exact power is
  correctly rounded. Every TEC, RMS and height node of the fifteen public GIMs
  for 2024 day 001 is the node RTKLIB's `readtec` holds. The writer states a
  value with a field the reader takes back as that value, so a product read from
  a file writes and reads back exactly. A value no such field reaches is stated
  as the field whose decimal is the value, and reads back one unit in the last
  place away: a product built from samples holding `0.7` was refused, since
  every field that fits `I5` gives `0x1.6666666666667p-1` where `0.7` is
  `0x1.6666666666666p-1`, and it is written and read back as that product.
- **Breaking.** An IONEX axis is read in the direction its step gives. IONEX 1
  gives an axis as "'LAT1' to 'LAT2' with increment 'DLAT'", which says nothing
  about the direction, so a file may run its latitudes south to north or its
  longitudes east to west; only a step whose sign contradicts its bounds is
  refused. `Ionex::lat_nodes_deg`, `Ionex::lon_nodes_deg` and the matching
  `TecGridSamples` fields hold the nodes in that order, the bracketing and
  clamping follow it, and the writer writes the axis records back as they were
  read. A file with an ascending latitude axis was refused for nodes "not
  strictly descending".
- An IONEX slant delay interpolates across the longitude seam of a grid that
  closes the circle. IONEX 1's example 1 runs its longitudes 0 to 355 by 5,
  covering every longitude without naming the seam twice; a query at 357.5 was
  refused as outside the coverage under the strict policy, and held at 355 under
  the hold policy, rather than interpolating between the node at 355 and the
  node at 0.
- An IONEX slant delay along a line of sight through a pole gives a value. The
  quotient that names the pierce-point longitude divides by the cosine of its
  latitude, which is zero at a pole, and the spherical-trig quotients can round
  past 1, where `asin` gives NaN; each is held inside its domain now.
- An IONEX `EXPONENT` an earlier map set stays in effect for the maps after it,
  and the map that inherits one reads at it and is reported as the new
  `IonexWarning::ExponentCarriedIntoMap`, naming the map, the exponent and the
  line that set it. IONEX 1 says of the header records that "Each value remains
  valid until changed by an additional header record", which carries an exponent
  across a map boundary; its 3-D example restates one at the start of a map
  rather than relying on the carry, so it neither needs nor contradicts the
  rule. Such a map was refused, which compounded: one map changing the exponent
  refused every later map that did not restate it. The writer still restates an
  exponent at a map start, as that example does and as RTKLIB, which reads only
  the header `EXPONENT`, needs.
- A data record inside an IONEX map holding a character outside ASCII is refused
  where it is read, naming the map, the band and the line. Such a record was
  split on whitespace instead, which can place its values at the wrong nodes
  where a field is also blank. Header text is unchanged, so the UTF-8
  `DESCRIPTION` records of `uhrg0010.24i` still read.
- IONEX reader messages count maps from 1, as a file numbers its own maps in its
  `START OF ... MAP` records, and the `INTERVAL` finding names the later map of
  the first pair spaced otherwise. A message named the first map "map 0".
- **Breaking.** `IonexHeader::maps_in_file` keeps the `# OF MAPS IN FILE` record
  a file carried, and the writer writes that value back. IONEX 1 counts every
  TEC, RMS and height map there, while CODE, IGS and UPC write the number of TEC
  maps: 25 with 25 RMS maps, 97 with 97. A product read from a file writes the
  count it came with; one built from samples writes the TEC map count, which is
  what those producers write.
- **Breaking.** `Ionex::to_ionex_string` returns `Result<String>` and refuses,
  naming it, a value, axis or header field that its IONEX field cannot hold
  exactly, and `scenario::ionex_content_fingerprint` returns
  `Result<String, ScenarioError>` because it hashes that text. The writer
  rounded every value to the header exponent without checking that it read back
  as the value held, wrote an axis step such as 0.25 degrees as 0.2, and wrote a
  field wider than its columns out of them. It now writes every value exactly.
  When one exponent does, the file has that one: the product's `EXPONENT` where
  it writes each TEC, RMS and height value as a whole number within `I5` other
  than the non-available marker `9999`, otherwise the nearest exponent that
  does, the finer of two equally near, which the product read back then has.
  Such a file has no `EXPONENT` record inside a map, so RTKLIB, which reads only
  the header `EXPONENT`, reads it too. RTKLIB places every node of a file that
  does carry those records at the right latitude and longitude, band splits
  included, but scales each value by the header exponent, so it reads the values
  of such a file as other numbers. Otherwise the header keeps the product's
  `EXPONENT`, an `EXPONENT` record before a band record gives each data block an
  exponent that writes it exactly, a latitude whose values no one exponent
  writes is split into band records over runs of its longitudes, and a map
  restates the exponent where the map before it left another in effect. Only a
  value that no exponent writes exactly is refused. A text field is written when
  it reads back as itself: within its columns, counted in bytes as the reader
  takes them, without control characters or the blanks the reader trims, so the
  UTF-8 `Hernández` in a `DESCRIPTION` of `uhrg0010.24i` is written back.
- **Breaking.** An IONEX slant delay maps vertical TEC to the line of sight with
  the single-layer `1/cos(z')` at the shell height whatever the product's
  `MAPPING FUNCTION` declares, and the new
  `IonexSlantDelayStatus::assumed_mapping` names the case a product declaring
  anything but `COSZ` falls in, as the new `IonexAssumedMapping`, while
  `IonexSlantRefusal::MappingFunction` carries the declaration itself, as the
  new `IonexMappingDeclaration`: `Declared(function)` for the record the product
  carries, `Absent` for a product with no record. IONEX 1 gives `COSZ` as
  `1/cos(z)`, `NONE` as "no MF used (e.g. altimetry)" and `QFAC` as "Q-factor"
  with no formula, and says others might be introduced. The CODE, ESA, JPL and
  UPC final GIMs for 2024 day 001 declare `NONE` and
  `EMR0OPSFIN_20240010000_01D_01H_GIM.INX` declares `MOD`, while their
  descriptions name the mapping function their maps were determined with, so a
  delay on them is computed and flagged rather than refused; it was computed
  with `1/cos(z')` before and carried no flag. The new
  `IonexMappingPolicy::Declared` maps only with the factor the product declares
  and refuses one whose code defines none, with the new
  `Error::IonexSlantUnavailable` carrying `IonexSlantRefusal::MappingFunction`.
  A product whose height maps give every node one and the same height uses the
  shell height `HGT1` plus that height, as IONEX 1 defines a node's height. One
  whose height maps give nodes different heights, or give a height as
  non-available, is refused with `IonexSlantRefusal::VaryingHeights` or
  `IonexSlantRefusal::HeightNotAvailable` under either mapping policy, since the
  delay uses one shell height.
- **Breaking.** `ionex_slant_delay_with_policy`, `ionex_slant_delay_results`,
  `Ionex::slant_delays_batch_results` and
  `ObservableIonosphereCorrection::IonexWithPolicy` take the new
  `IonexSlantPolicy`, which holds an `IonexCoveragePolicy`, the new
  `IonexMissingNodePolicy` and the new `IonexMappingPolicy`; an
  `IonexCoveragePolicy` converts into it. `IonexSlantDelayStatus` is a struct:
  `held` holds the coverage miss a hold policy held the value through,
  `degraded` the non-available nodes a renormalizing policy interpolated around,
  and `assumed_mapping` which case a product falls in where the single-layer
  factor mapped it and it declares anything but `COSZ`, as the new
  `IonexAssumedMapping`: `NoMapping`, `QFactor`, `Other` or `Absent`. That names
  the case without the code's text, so the status stays `Copy` and a batch
  allocates nothing per ray; the text of an `Other` code is in
  `IonexHeader::mapping_function`, as `IonexMappingFunction::code`.
  `IonexSlantDelayStatus::VALID` replaces `IonexSlantDelayStatus::Valid` with
  its three fields `None`, and a status with `held: Some(error)` replaces
  `IonexSlantDelayStatus::Held(error)`. `is_valid` is true when `held` and
  `degraded` are both `None`, which is to say the value is the one the grid
  gives at the request. It does not read `assumed_mapping`, which the CODE, ESA,
  JPL, UPC and EMR final GIMs all carry: the factor applied is the single-layer
  one either way, and a caller that will not take it on such a product reads
  that field or uses `IonexMappingPolicy::Declared`.
- IONEX slant-delay refusals count maps from 1 too: `IonexMissingNodes` carries
  `map_number` in place of `map_index`, and so do
  `IonexSlantRefusal::VaryingHeights` and
  `IonexSlantRefusal::HeightNotAvailable`. A message named the first height map
  "height map 0". The cell indices such a message gives stay 0-based, since a
  map's nodes are an array a file does not number. `IonexMissingNodes` carries
  the cell's other column as `lon_index_next`, which is `0` on the cell that
  closes the circle rather than `lon_index + 1`: a grid running 0 to 355 by 5
  has no column 72, and a message named a missing node there `[lat][72]`, which
  a caller indexing the longitude axis with it could not read.
- `Error::IonexNodesNotAvailable` carries its `IonexNodeGap` in a `Box`. The gap
  names two cells of four nodes each and is 80 bytes, which sat inline in the
  crate's `Error`, and so in every `Result` the library returns: `Error` is 32
  bytes with it boxed where it was 80 without. `sidereon-scoreboard`'s error
  wraps `Error` and had reached the 128 bytes at which `clippy::result_large_err`
  fires; it is 120 again. The allocation falls only where a query weights a
  non-available node, and `IonexSlantDelayStatus` holds the gap directly rather
  than through `Error`, so it stays `Copy`.
- `IonexMissingNodePolicy::Renormalize` gives an IONEX slant delay whose
  interpolation weights non-available nodes a value from the weighted nodes that
  hold values, their bilinear weights renormalized to sum to one, and from the
  weighted maps that give a value, their temporal weights renormalized the same
  way, and marks the value degraded. With no weighted node holding a value there
  is still no value.
  `IonexMissingNodePolicy::Strict`, the default, refuses.
- **Breaking.** `TecGrid::new` takes `Option<f64>` values, `None` marking a node
  without a value. A grid could not hold such a node, so a caller building one
  from IONEX maps with non-available nodes had to put a number there. A query
  that weights such a node returns the new `TecGridError::NodesNotAvailable`, and
  the new `TecGrid::vtec_at_pierce_point_with_policy`,
  `regular_tec_xyz_with_policy` and `regular_tec_grid_delay_xyz_with_policy` take
  an `IonexMissingNodePolicy` and return a `TecGridEvaluation` whose `degraded`
  field marks a renormalized value, renormalized within each bracketing epoch and
  then in time, as for IONEX products.
- **Breaking.** An IONEX value a file gives as `9999` is `None`. IONEX 1 says
  "Non-available TEC values are written as '9999'", and that RMS and height
  values are "formatted exactly in the same way". `Ionex::tec_maps`,
  `Ionex::rms_maps`, `TecGridSamples::tec_maps`, `TecGridSamples::rms_maps`,
  `TecSample::vtec_tecu` and `TecSample::rms_tecu` hold `Option<f64>`. A `9999`
  field was read as 999.9 TECU at the default exponent, and the slant delay
  interpolated it; `EMR0OPSFIN_20240010000_01D_01H_GIM.INX` gives TEC map 21 at
  latitude 10.0, longitude 145.0 as `9999`. A slant delay whose interpolation
  weights such a node is refused with the new `Error::IonexNodesNotAvailable`,
  whose `IonexNodeGap` names the map, the cell and the missing nodes as
  `IonexMissingNodes`. A node or map whose interpolation weight is zero is not
  used, so a query on an available node, or at a map's epoch, still has a value.
  A TEC map the file gives no RMS map for has an RMS map of `None`, and RMS maps
  without a value at any node are dropped.
- **Breaking.** IONEX `START OF HEIGHT MAP` blocks are read into the new
  `Ionex::height_maps`, `TecGridSamples::height_maps` and
  `TecSample::height_offset_km`, in kilometers. The value a height map holds at a
  node is added to `HGT1` to give the single-layer height there, rather than
  being a shell height of its own: IONEX 1's example 1 gives every height as `0`
  with `HGT1` at 400 km. A height map without a value at any node is kept, since
  it says the heights are unknown. A height map's bands used to be added to
  whichever grid was open, and its `END OF HEIGHT MAP` record then failed to
  read as a value.
- **Breaking.** An IONEX product keeps its descriptive header records in the new
  `IonexHeader`, from `Ionex::header` and in `TecGridSamples::header`: the version
  and satellite system, `PGM / RUN BY / DATE`, the `DESCRIPTION` and `COMMENT`
  records, `INTERVAL`, `MAPPING FUNCTION` as the new `IonexMappingFunction`,
  `ELEVATION CUTOFF`, `OBSERVABLES USED`, `# OF STATIONS` and `# OF SATELLITES`.
  A file without one of them reads as the value `IonexHeader::new` gives it.
  `Ionex::from_node_samples` takes the header as its last argument. The writer
  writes these records back, and writes `EPOCH OF FIRST MAP`,
  `EPOCH OF LAST MAP`, `# OF MAPS IN FILE`, `MAP DIMENSION` and `END OF FILE`
  from the maps; it wrote none of them. `ELEVATION CUTOFF` describes the data the
  maps were determined from and does not limit the elevations a slant delay is
  evaluated at.
- `Ionex::parse_with_warnings` and `Ionex::parse_str_with_warnings` return the
  product with `IonexWarning`s for header records that summarize the maps and
  disagree with them or are absent: an `EPOCH OF FIRST MAP` or
  `EPOCH OF LAST MAP` naming another epoch than the maps carry, a
  `# OF MAPS IN FILE` counting neither the TEC maps nor every map, a nonzero
  `INTERVAL` that is not the spacing of the maps, an `IONEX VERSION / TYPE` that
  is not the first record, and a missing record IONEX 1 marks mandatory,
  `END OF FILE` included. These are reported rather than refused because every
  map carries its own epoch and bands, so the values read do not depend on them.
  `UPC0OPSFIN_20240010000_01D_02H_GIM.INX` gives an `EPOCH OF LAST MAP` of
  2024-01-01 23:59:24 for a last map at 2024-01-02 00:00:00, and
  `IGS0OPSRAP_20240010000_01D_02H_GIM.INX` has no `END OF FILE`. A summary
  record that cannot be read is skipped and counted in `skipped_records`, and a
  whole-valued decimal reads as its whole number, as in the `0.00` seconds and
  `1800.0` interval of `CAS0OPSFIN_20240010000_01D_30M_GIM.INX`.
- **Breaking.** `SYS / PHASE SHIFT` records are read in their columns,
  `A1,1X,A3,1X,F8.5,2X,I2.2,10(1X,A3)`, or by their fields where a record is not
  laid out in them, and written in those columns at every version. Satellite
  lists continue past ten satellites on `18X,10(1X,A3)` records, in the file
  header and after an event, and the writer writes them ten to a record. A list
  of more than ten was refused as a count mismatch, and a list read on one
  record was refused as too wide to write. A record not in its columns is read
  by the reading its fields agree with: after the code, a number is a
  correction followed by a satellite count, or a count after a blank
  correction, and the reading whose count agrees with the satellites after it
  is taken; a lone whole number, which reads both ways, is refused as
  ambiguous. A satellite a list names by a well-formed designator that
  `GnssSatelliteId` does not hold, such as `R28`, is kept as written in the new
  `ObsPhaseShift::unrepresentable_satellites`, written back in its list,
  counted in `skipped_records` and given no correction; a record naming only
  such satellites is not one for every satellite, as the new
  `ObsPhaseShift::covers_every_satellite` says, and contradictions and later
  blocks key it by its designator. A token naming no satellite is still
  refused. A record naming only its constellation, which RINEX 3.05 section
  5.2.12 gives where "the applied phase corrections or the phase alignment is
  unknown" ("the observation code field and the rest of the SYS / PHASE SHIFT
  header record field of the respective satellite system(s) are left blank"),
  is read, kept and written back, `ObsPhaseShift::code` being
  `Option<String>`; the IGS headers of JOZ200POL, NKAY00LAO, POLV00UKR,
  POVE00BRA, SCTB00ATA and TOAY00LAO for 2026 day 001 carry such records and
  were refused. The IGS headers of BADG00RUS and ARHT00ATA for 2026 day 001, each
  naming R28 in four lists, were refused, for a list longer than ten.
  `ObsPhaseShift::correction_cycles` is
  `Option<f64>`, `None` where the correction is blank, which RINEX 4.02 gives as
  "Correction applied (cycles) or blank if none"; a record with a blank
  correction before its satellite count was refused, its first satellite read
  as the count. A correction its `F8.5` field cannot hold is refused where it is
  read, and the writer refuses a product holding one; the writer used to write
  such a correction in exponent form.
- **Breaking.** The header records in one block, the file header or one event's
  records, apply whatever their order. RINEX "allows the free ordering of the
  header records" (3.05 section 5.2.1) and does not say how overlapping records
  in one block combine; the rules here are this reader's policy, chosen to be
  consistent with that free ordering, not rules RINEX states. A
  `SYS / PHASE SHIFT` record naming a satellite applies to it over the record
  for every satellite of its code, as it did, and a `SYS / SCALE FACTOR` record
  naming a code applies over the record for every code of its constellation,
  where the last covering record applied. A block giving a key that changes how
  observations are read two values is refused as contradictory, values
  compared as numbers so `-0.0` and `0.0` are one value: two factors for one
  code or for every code of a constellation, or two channels for a GLONASS
  slot. Phase shifts and GLONASS biases change nothing an observation reads
  as, so a block giving a satellite's code two corrections, or a GLONASS code
  two biases, is read: the records are kept and written back, each
  contradiction is counted in `skipped_records`, and that signal reads as the
  new `CorrectionUnavailable::Ambiguous`. The IGS header of MET300FIN for 2026
  day 001, which gives C02 and C05 L2I and L6I corrections of both 0 and 0.5,
  was refused. A blank `GLONASS COD/PHS/BIS` record beside biases in one block
  is still refused, as the header holds no blank record beside biases. Two records of a label holding one value that read as
  different values are refused the same way, naming the label: `INTERVAL`,
  `MARKER NAME`, `MARKER NUMBER`, `MARKER TYPE`, `APPROX POSITION XYZ`,
  `ANTENNA: DELTA H/E/N`, `ANT # / TYPE`, `REC # / TYPE / VERS`,
  `OBSERVER / AGENCY` and `SIGNAL STRENGTH UNIT` in any block, and
  `RINEX VERSION / TYPE`, `TIME OF FIRST OBS`, `TIME OF LAST OBS`,
  `LEAP SECONDS` and `# OF SATELLITES` in the file header. Identical duplicates,
  and records whose text differs but which read as one value, are read; the
  later of two records used to apply. `PGM / RUN BY / DATE`, `COMMENT` and
  `PRN / # OF OBS` are read as they were. From RINEX 4.00, which says of `SYS / PHASE SHIFT` and
  `GLONASS COD/PHS/BIS` that "the lines should be ignored by RINEX decoders and
  encoders", `CarrierPhaseRow::phase_shift_cycles` is 0 whatever those records
  say, and they are not refused as contradictory; they are kept and written
  back.
- **Breaking.** `CarrierPhaseRow::phase_shift_cycles` is
  `Result<f64, CorrectionUnavailable>`: `Ambiguous` where one header block gives
  the satellite's code different corrections, and `Unknown` where the only
  record covering it names just its constellation. `GLONASS COD/PHS/BIS`
  records are read in their columns, `4(1X,A3,1X,F8.3)`, or by their fields
  where a record is not laid out in them, and written in those columns;
  `ObsHeader::glonass_cod_phs_bis` holds `Option<f64>` biases, a blank bias
  kept and written back blank. The IGS header of NICO00CYP for 2026 day 001,
  whose C1P bias is blank, was refused, the code after it read as the bias.
  The new `ObsHeader::glonass_code_phase_bias` gives a GLONASS signal's bias:
  `Unknown` for a blank record or bias, and `Ambiguous` where a block gives the
  code two.
- **Breaking.** The header records an event epoch carries take effect for the
  epochs after it. RINEX has allowed header records after every event flag
  since 2.10, and flag 4 says "header information follows"; they were kept as
  text and never applied, so a flag 4 epoch declaring a longer, reordered or
  shorter type list left the epochs after it read by the old list, with values
  unread, under the wrong codes, or blanks read as missing values, and a scale
  factor, phase shift, GLONASS channel, interval, marker, position or antenna an
  event declared was never in effect. Every such record after flags 2 to 5 is
  now read as the file header reads it, and refused as the file header refuses
  it: `SYS / # / OBS TYPES` at version 3 and `# / TYPES OF OBSERV` at version 2,
  with continuations, `SYS / SCALE FACTOR`, `SYS / PHASE SHIFT`,
  `GLONASS SLOT / FRQ #`, `GLONASS COD/PHS/BIS`, `INTERVAL`,
  `MARKER NAME/NUMBER/TYPE`, `APPROX POSITION XYZ`, `ANTENNA: DELTA H/E/N`,
  `ANT # / TYPE`, `REC # / TYPE / VERS`, `OBSERVER / AGENCY` and
  `SIGNAL STRENGTH UNIT`. The records stay verbatim in `special_records`; any
  other record takes no effect.
- **Breaking.** `ObsHeader::obs_codes` is the union of every list the file
  declares for each constellation, the file header's codes first, then codes
  declared later in the order first declared, and every epoch's observations
  and cycle slips are index-aligned to it, blank under a code the list in effect
  at the epoch does not declare. The new `ObsHeader::declared_obs_codes` holds
  the lists the header itself declares. Scale factors apply to the values read
  after them, and values stay physical.
- `RinexObs::header_at` returns the header in effect at an epoch: the file
  header with every event at or before it laid over it, `obs_codes` the union
  and `declared_obs_codes` the lists in effect. `RinexObs::header_timeline`
  builds every header once, as an `ObsHeaderTimeline` a loop over the epochs
  looks each up in. An event's records are one header block laid over the header
  in effect, by this reader's policy, chosen to be consistent with "Each value
  remains valid until changed by an additional header record" (section 6.5): a
  value replaces the one in effect, a type list replaces its constellation's
  list, and a block's complete declarations for one constellation add to one
  another as the file header's do. RINEX gives each constellation one
  declaration, so adding a repeated one is this reader's policy too. A phase
  shift or scale factor for every
  satellite or code replaces the earlier records for its code or constellation,
  and one naming satellites or codes replaces those only; a GLONASS slot or bias
  replaces that slot's or code's, and a blank `GLONASS COD/PHS/BIS` record
  replaces every bias. One event's records give the header the same records give
  split across consecutive events wherever no two of them give one key a value,
  and `obs_codes` holds every code a declaration names however the declarations
  are grouped. Both refuse a product whose event record does not read.
- **Breaking.** `RinexObs::to_rinex_string` writes a product whose events
  declare header records back whole: the file header's own lists from
  `declared_obs_codes`, each event's records verbatim, and each epoch by the
  code lists or version 2 names and the scale factors in effect at it, its
  values placed by code from the union. The strict read-back compares
  `declared_obs_codes` too. It refuses, with new variants of
  `RinexObsWriteError`, a product whose `obs_codes` is not the union of the
  lists its header and events declare (`CodeListsNotUnion`), a value or cycle
  slip under a code the list in effect at its epoch does not declare
  (`ValueOutsideDeclaredList`), and an event record that does not read
  (`EventRecordsUnreadable`), and a version 2 product holding a declared list its type names do not state (`DeclaredListNotStated`), which the comparison used to rebuild on both sides; a version 3 `PRN / # OF OBS` count past the codes
  the file header declares, and a value past its constellation's union, are
  refused by name with `CountsWithoutCodes` and `ValuesWithoutCodes`. At version
  2, `ScaleFactorsInVersionTwo` counts the scale factors events declare, and a
  product whose events declare type names is written with header names reading
  as its declared lists.
- **Breaking.** `RinexObs::downgrade_to_rinex2` downgrades a product whose
  events declare code lists or type names. Each stretch of epochs sharing the
  lists in effect is laid out for version 2 as the file header's lists are, and
  its per-list changes are reported inside the new
  `ObsDowngradeChange::InEventLists`, naming the event's epoch. Each event's
  type records become the version 2 `# / TYPES OF OBSERV` records its stretch
  lays out as; the file header takes the names its own lists lay out as, also where an event before the first epoch declares others, and the values
  are held under the union of what every stretch's names read as. An event's
  `SYS / SCALE FACTOR` records are removed as the file header's are, and a
  version 2 type record a version 3 event carried without effect is removed,
  since a version 2 reader applies it. Each event whose records change is
  reported as the new `ObsDowngradeChange::EventRecordsRewritten`, with its
  records before and after. A value the list in effect at its epoch does not
  declare is refused with `RinexObsWriteError::ValueOutsideDeclaredList`, as the
  writer refuses it. A product read from RINEX 4.00 or later loses its
  `SYS / PHASE SHIFT` and `GLONASS COD/PHS/BIS` records, from the file header
  and from its events, each removal reported as the new
  `ObsDowngradeChange::DeprecatedRecordsRemoved` naming the label and where the
  records were: version 4 declares them to be ignored, and a version 2 file
  would apply them. Records contradicting each other, which version 4 does not
  refuse, no longer make the downgrade fail.
- **Breaking.** `carrier_phase_rows` takes the header in effect at the epoch
  rather than the product, since a phase shift or GLONASS channel an event
  declares applies to the epochs after it.
- Positioning seeds each epoch from the `APPROX POSITION XYZ` and GLONASS
  channels in effect at it, and the RTK arc builders read carrier frequencies by
  each side's GLONASS channels in effect. A satellite whose carrier changes, as a
  GLONASS slot re-declared on another channel changes it, starts a new ambiguity
  id, `<satellite>~freq<n>`, with its own wavelength, where one wavelength used to
  scale every epoch, also where the change falls in an epoch left out of the
  arc. At each epoch a satellite's measurement is formed from the first
  configured pair whose code and carrier phase are both present, and its
  ambiguity is named by the carriers that measurement is on: a measurement on
  another carrier phase observable or frequency, including a return to an
  earlier one, starts a new ambiguity, and a fallback to another code on the
  same carrier keeps it. Each receiver's carriers are followed over every epoch
  wherever their phase is recorded, whether or not a code was, so a frequency
  change at an epoch left out still starts a new ambiguity, and a loss of lock
  at an epoch left out is set on the carrier's next measurement. Splitting arcs at cycle slips splits them at a carrier change too, and a
  split id, `<id>@<receiver>#<segment>`, is built from the carrier's ambiguity
  id and keeps its wavelength, where it was built from the satellite id, ran
  across the change and took the satellite's first wavelength. Fixing and
  holding look a derived id's wavelength and offset up by its own entry, its
  reference-qualified entry `<id>|ref=<reference>` as the ionosphere-free
  preparation keys them, then its carrier arc's, before the satellite's; the
  dual-frequency pipeline refused a split arc with a missing wavelength and
  scaled a later carrier arc by the first. Where the reference satellite's
  single-difference ambiguity is not its first arc, the dual-frequency pipeline
  names each other satellite's ionosphere-free ambiguity by the reference arc
  too, `<id>|ref=<reference>`, since its offset holds the double-difference
  wide-lane integer; a change of the reference's ambiguity made the pipeline
  refuse the arc for want of an offset. `solve_rtk_arc` refers the filter to
  the reference ambiguity each epoch carries: where the reference satellite's
  ambiguity arc changes, at a slip or a carrier change, the integers held
  against the old one are released and fixed again against the new one, where
  the arc was refused with `UpdateError::ReferenceChanged`. `update_epoch`
  still refuses a state carried across such a change. A code a list declares twice is read
  by its first copy, blank or not, in the RTK builders and in observation QC,
  where RTK read the last and QC the first holding a value; a satellite the two sides track on different carriers is
  left out of that epoch, as no integer single difference exists for it. Observation QC expects only the codes
  in effect at each epoch, picks each band's code by the order of the list in
  effect there rather than the union's, reads GLONASS channels in effect, and judges a gap by
  the `INTERVAL` in effect at the later epoch; multipath arcs do the same. Lint
  compares each `INTERVAL` with the spacing of the epochs it is in effect for,
  judges gaps within that spacing, and takes a GLONASS slot an event declares.
  Repair sets `INTERVAL` from the epochs before an event declaring another,
  counts `PRN / # OF OBS` only for codes the file header declares, and keeps the
  header records an event carries when dropping unsupported records. An event
  record that does not read is the new lint finding
  `Finding::ObsEventHeaderUnreadable` and the new QC note
  `ObservationQcNote::EventHeaderRecordsUnread`.
- **Breaking.** `ObsEpoch::epoch` is `Option<ObsEpochTime>`, `None` for an
  event whose epoch fields are blank. RINEX 2.11 and 3.05 let an event without
  a significant epoch leave them blank, and both specifications' example files
  do; the reader refused every such record, so it could not read those files.
  A version 3 record is read in its columns or loosely, as `>`, the flag, the
  count and an optional clock offset, and a version 2 record as the flag and
  count after 28 blank columns, read from their columns so a count of 100 to
  999, which abuts the flag, is read too. Blank epoch fields on an observation or cycle
  slip record, or with picoseconds, are refused. The writer writes a blank
  event's epoch fields blank, and refuses an observation or cycle slip epoch
  with no time with the new `RinexObsWriteError::EpochTimeMissing`. Positioning,
  observation QC and repair skip an observation epoch with no time, as they
  have no time to place it at.
- The RINEX RTK arc builders index rover epochs by time over observation
  epochs only. Every rover epoch was indexed, so an event or cycle slip epoch
  sharing an observation epoch's time replaced that epoch in the index, and
  the base epoch it matched was skipped for want of rover observations.
- **Breaking.** `ObsEpoch` gains `cycle_slips`, the cycle slips a flag 6 epoch
  reports, per satellite and index-aligned to `ObsHeader::obs_codes` as `sats`
  is. RINEX writes them in the observation record layout, with the slip in
  place of the observation, and counts satellites on the epoch line, listing
  them there at version 2. Every flag above 1 was read as an event whose count
  was a number of verbatim records, so a version 3 file's slips were kept as
  text no consumer read, and a version 2 slip epoch whose records ran past one
  line, or whose satellite list continued, left lines behind to be read as
  epochs. Flag 6 records are now read exactly as observation records are, in
  both layouts, and written back from `cycle_slips`; the strict read-back
  compares them. `sats` and `special_records` stay empty for a slip epoch, so
  positioning, observation QC and repair, which skip epochs flagged above 1,
  never take a slip for a measurement: `TIME OF LAST OBS` and
  `PRN / # OF OBS` count observation records only. Lint no longer reports a
  slip epoch as an event (OBS-B07), and compares its declared count with the
  satellites whose slips it holds. `CYCLE_SLIP_FLAG` names the flag.
- **Breaking.** `RinexObs::downgrade_to_rinex2` moves cycle slips with their
  codes as it moves observations, refuses slips past their constellation's
  codes with `RinexObsWriteError::ValuesWithoutCodes`, and reports a slip
  rounded to three decimals as the new `ObsDowngradeChange::CycleSlipRounded`.
  A constellation named only by slips has its list stated in a version 2 file,
  as a reader builds one for it.
- **Breaking.** `RinexObs::to_rinex_string` returns
  `Result<String, RinexObsWriteError>`, as `RinexClock::to_rinex_string` already
  did, and returns text only when reading it back gives the product: every
  header record, code, value, indicator and event record, compared field by
  field. It used to drop, round, wrap or truncate whatever a file could not
  carry and return the text anyway, so a caller could not tell a faithful file
  from one that said something else. The error names the first field that would
  change and its value before and after. Diagnostics about the text a product
  was parsed from - unretained header labels, skipped records, the count an
  epoch line declared - are not part of what is written and are not compared.
  `Scenario::to_rinex_string` returns the same `Result`.
- **Breaking.** A version 2 product is written only when each constellation's
  code list is exactly what one list of version 2 names reads as, which is true
  of every file read at version 2. Any other product is refused with
  `RinexObsWriteError::CodeListsNotVersionTwo`, naming the constellation and
  position no name fits. The list is found a position at a time, and because
  what a constellation reads at a position depends only on the codes it holds
  before it, that search finds a list whenever one exists. A version 2 product
  carrying `SYS / SCALE FACTOR` records is refused with
  `RinexObsWriteError::ScaleFactorsInVersionTwo`: this reader applies them, but
  version 2 readers that do not know the record, RTKLIB among them, read the
  scaled numbers as physical ones. A version 2 product holding a code list for
  a constellation that no observation or `PRN / # OF OBS` count names, and that
  the version record of a file with no observations does not name, is refused
  with `RinexObsWriteError::CodeListNotStated` unless the type names read as
  it: a reader builds no list for it, so the file would lose it. The names are
  chosen to read as the largest set of such lists they can, so GPS `C2W` and
  GLONASS `C2P` with no observations are written at 2.12 as `P2`, which reads
  as both, rather than `C2`, which reads as only the GPS one.
- `RinexObs::downgrade_to_rinex2` turns any product into one a version 2 file
  can state exactly, and returns every change it made as an
  `ObsDowngradeChange`: a code renamed to what its column reads as, a code moved
  to another position in its list, a blank code added so every constellation's
  list matches, a code list the version 2 file does not state removed (a
  version 3 product with no observations is named for a constellation whose
  list no count keeps, GPS first, so the file keeps every list it can),
  `SYS / SCALE FACTOR` records
  removed from values that are already physical, a receiver clock offset
  rounded to the nine decimals a version 2 epoch record holds, a value rounded to the three
  decimals a field holds once no factor scales it, and epoch picoseconds
  version 2 has no field for. Values and `PRN / # OF OBS` counts move with their
  codes. The result is returned only once it writes and reads back exactly, and
  what version 2 still cannot state - more than 999 observation types
  (`RinexObsWriteError::TooManyObservationTypes`), a year outside the two-digit
  window - is refused. It renames only codes no
  version 2 layout can keep: a code no version 2 name spells, or a second copy
  of a code its constellation already holds. Whether a canonical code is kept
  does not depend on column order, since the reader gives it at the first column
  whose name reads as it, so naming a column for every code any name reads as
  keeps all the rest. A second copy of a held code gets a column named with its
  own spelling rather than one that reads as a different tracking attribute,
  shared with another constellation's second copy under the same name wherever
  it reads the same there, and a code no name spells shares a column already
  reading as its spelling's code. A name kept as written adds a column only when
  no column already reads as it. Columns are ordered to move the fewest codes
  any order of them can: keeping the most codes in place is an assignment of
  columns to positions, and the reading rule's ordering - a group of names read
  as one code gives it to its first column - is met by branch and bound over
  those assignments. Every assignment solved is also repaired into an order
  keeping the rule, and the best order seen is kept however the search ends.
  The search is limited to 100 million steps counted inside the assignment
  method; when that does not prove an order best, the best distinct orders it
  saw, and columns sorted by the earliest position they keep a code at, are
  improved with the remaining steps by moving one column, swapping two, or
  moving three around a cycle, while that, or its repair, keeps the rule and
  more codes in place. Every step of the search and of improvement is counted,
  so the total work is bounded, and a search that finishes never drops a valid
  assignment for want of work to check it, so what it proves best is. The
  layout's columns are built from indexes of which column first reads as each
  code, rather than by reading the names again for every question, and match
  the previous construction column for column. A layout wider than the 999 types a header
  declares is refused before any of it. The result still reads back exactly
  with every move reported. Values or counts past their constellation's
  codes are refused (`ValuesWithoutCodes`, `CountsWithoutCodes`) rather than
  dropped. A product version 2 already states comes back unchanged.
- **Breaking.** `ObsEpoch` carries `special_records`, the records an event epoch
  was followed by, in place of `special_record_count`. An event epoch is
  followed by header or comment records - a flag 3 epoch by the marker, antenna
  and position of a new site occupation - and those were counted and thrown
  away. The epoch was then written back declaring zero, so a file that changed
  site lost what it changed to, and nothing said so. They are kept as they were
  written, and written back under their own count.
- **Breaking.** The `OBS-B11` quality-control finding is gone. It reported that
  an event epoch's records were not retained, which is no longer true of any
  file. Repairing a file that has them no longer fails: it used to refuse rather
  than drop them, and now it carries them. `drop_unsupported` still drops them
  and still reports having done so.

### Fixed

- A TDM line whose keyword is `COMMENT` is a comment or nothing. CCSDS
  503.0-B-2 4.2.5 c) excepts `COMMENT` from the KVN syntax, and 4.5.3 requires
  at least one space after the keyword, so `COMMENT=value` is neither a comment
  nor an assignment; `tdm::parse_kvn` read it as a field keyed `COMMENT` and now
  refuses it with `TdmError::MalformedLine`. `tdm::encode_kvn` wrote such a field
  as `COMMENT = value`, which reads back as a comment whose text is
  `= value`, so a message carrying one parsed, encoded and reparsed to a
  different value, one comment longer and one field shorter. The writer refuses
  such a field with `TdmError::KeywordNotAssignable`, which `validate_tdm` raises
  for a value a caller built, since the reader no longer produces one. The
  scheduled fuzz run on the `tdm_round_trip` target found the parse-side defect
  on a 1071-byte input whose line 112 is `COMMENT=`; that input is kept as a
  regression fixture, and now meets the character-set rule at line 2 before it
  reaches line 112, so it pins that refusal and that line 112 still holds the
  construct. The minimal messages in `a_comment_keyed_assignment_is_refused_in_both_sections`
  pin the `COMMENT` assignment itself.
- The IONEX writer lays records out as IONEX 1 defines them: axes in
  `2X,3F6.1`, band records in `2X,5F6.1`, values in `16I5`, and `EXPONENT` only
  where it is not `-1`. Axis and band records were written in `8.1` fields, which
  a reader that takes those records by column, as RTKLIB does, reads as other
  numbers, and values were written as a blank and four columns, so a five-digit
  value broke the columns.
- IONEX values are read in their `16I5` columns, so two adjacent five-digit
  values read as two values, not one. A data record not laid out in those
  columns is read by its whitespace-separated fields. One laid out in them with
  an empty field is refused, since no reading of it can tell which value is
  absent, and a data record outside a band is refused rather than passed over.
- Each IONEX `LAT/LON1/LON2/DLON/H` record places its values by its own
  latitude, its longitudes `LON1 + DLON * m` and its height, read in its
  `2X,5F6.1` columns, or by its fields where it is not laid out in them. The
  record was not read and bands were placed in the order they came, so a band out
  of latitude order, or with another longitude range, was misplaced. A band whose
  latitude, longitude or height is not a node of the header grid is refused
  naming the coordinate, as is a map that gives a node twice or leaves one
  without a value. Bands may come in any order, and one latitude may span several
  bands.
- An IONEX `EXPONENT` record inside a map sets the unit of the data blocks after
  it, where the spec's 3-D example places it; the header `EXPONENT`, or `-1`
  without one, applies before it. Values are stored in TECU. Such a record before
  a map's first band was ignored, scaling the map by the wrong power of ten, and
  one between bands failed to read as a value. A map that gives no `EXPONENT`
  after an earlier map left another in effect is refused, because the spec does
  not say whether an exponent set in one map carries into the next.
- An IONEX `MAP DIMENSION 3` product, or one whose `HGT1 / HGT2 / DHGT` gives
  more than one height, is refused naming the record. Its TEC values are layer
  contributions, electron density times `DHGT`, which no single-layer map, sample
  set or slant delay here holds or uses. Such a file failed as a latitude band
  count mismatch.
- IONEX `START OF TEC MAP` records must number the maps 1, 2, 3 in order, and
  each `END OF ... MAP` record must give the number of the map it closes. An RMS
  or height map must give the number of a TEC map, once, and an
  `EPOCH OF CURRENT MAP` it gives must be that TEC map's epoch. None of this was
  checked, and RMS maps were paired with TEC maps by position.
- An IONEX epoch at hour 24 with a zero minute and second reads as 00:00 of
  the next day, in `EPOCH OF CURRENT MAP`, `EPOCH OF FIRST MAP` and
  `EPOCH OF LAST MAP`. `uqrg0010.24i` gives its 97th map, at `INTERVAL 900`, as
  `2024 1 1 24 0 0`, and the whole file was refused. Hour 24 with a nonzero
  minute or second is still refused.
- An IONEX value field reading `nan`, which IONEX 1 does not define, reads as
  non-available and is reported as `IonexWarning::NotANumberValue`, naming the
  map, the line and the node. `uqrg0010.24i` gives RMS map 55 at latitude 40.0,
  longitude -125.0 as `nan`, and the whole file was refused.
- An IONEX `AUX DATA` block in the header, where IONEX 1 defines it and real
  products carry their code biases, counts in `skipped_records` as one after the
  header did; it was passed over uncounted, as were unrecognized header records,
  which count too. An `AUX DATA` block that is not closed is refused, as are a
  file type other than `I` in `IONEX VERSION / TYPE` and two records of one
  header label that read as different values.
- CRINEX compression refuses a RINEX 2 cycle slip epoch that CRINEX 1 cannot
  carry. CRINEX 1 copies exactly as many lines after an epoch flagged above 1
  as its count, as RNX2CRX and CRX2RNX do, but a flag 6 epoch's count is
  satellites: a list continued past twelve of them, or records continued past
  five observation types, by the count in effect after any event that declared
  one, occupy more lines. Those lines were copied short, and the rest read as
  epochs, so the stream could not be read back. Such an epoch is refused with
  the lines it occupies; one within the limit is carried as before, and a
  RINEX 3 slip epoch, one record line per satellite, is unaffected. Expansion
  wrote every event's epoch line only to its count field, which dropped a slip
  epoch's satellite list and a timed event's clock offset and RINEX 4.02
  picoseconds. Every event's epoch line is now compressed and expanded whole,
  trimmed on the right, as RNX2CRX and CRX2RNX 4.2.0 copy it; an event's
  picoseconds take no part in the clock line's picosecond carry, as in CRX2RNX.
- A `LEAP SECONDS` record with a blank field before a written one keeps each
  field in its own six columns. The blank field was left out, so a record with
  no future count but a week and day was written with the week where the future
  count goes and the day where the week goes.
- GPS `C2` is the L2C pseudorange below version 2.12 and the L2P(Y) pseudorange
  from 2.12, which gave L2C its own names. Version 2 section 10.1.1 added it as
  "Observation code for L2C pseudorange (C2)", and RINEX 3 spells L2C `C2S`,
  `C2L` or `C2X` by channel. It was read as `C2C`, which is L2 C/A, at every
  version, so a file's L2 measurement came back named as a signal it was not.
- A tracking attribute is only claimed where version 2 names one. Galileo has a
  channel on every band and version 2 names none of them, and the same is true
  of QZSS L2C, so those read as the combined channel rather than asserting a
  single one the file never stated.
- The version 2 observation code table holds what RINEX 2.11 defines and
  nothing else. It used to carry the legacy differential-code-bias labels for
  Galileo and BeiDou, where `P1` and `P2` mean the first and second frequency
  whatever the constellation. Those belong to bias files, and the bias reader
  keeps them; version 2 says "P: Pseudorange GPS and Glonass: P code", and gives
  Galileo `C1`, `C5`, `C6`, `C7` and `C8`, whose digits already are the bands E1,
  E5a, E6, E5b and E5a+b. So a Galileo `C2` was read as E5a data, and a
  conforming `C5` and an invented `C2` both claimed that band. A Galileo `C5Q` is
  now written `C5`, losing the tracking attribute version 2 cannot carry, rather
  than a `C2` no reader defines.
- A BeiDou version 2 observation code names the band its digit does, whatever
  its kind. RINEX 2.11 has no BeiDou at all, so a BeiDou code there is an
  extension, and version 2 numbers its digits by frequency slot across every
  constellation rather than by a per-constellation count: B1I is slot 2, which
  RINEX 3 numbers band 2, while B2I and B3I are slots 7 and 6, which RINEX 3
  numbers the same. `C2` was read as B2I, which is a different signal, and the
  digit was remapped only for `C`, so a file's `C1` read as B1I while its `L1`
  read as B1C: one measurement pair read as two signals, with the phase on a
  band its own pseudorange did not use. Writing a code back reverses the
  numbering, so dropping a tracking attribute version 2 cannot carry no longer
  also moves the band.
- No constellation but GPS and GLONASS is given a `P` observation code. Version
  2 says "P: Pseudorange GPS and Glonass: P code". A mixed product could put a
  Galileo `P1` in a header, which no reader defines. A shared column is held to
  the same rule for every constellation in it, not only the one whose codes the
  name came from: a Galileo `C1X` reads `P1` back correctly, and that is not
  enough to put it under one.
- A version 2 `# / TYPES OF OBSERV` code wider than the two characters its field
  holds is rejected. The record is `9(4X,A2)`, so a longer token means the line
  is not in that layout. Such a code was kept whole, written back into a
  two-character field, and read again as a different code. This is stricter than
  readers that take the two columns and ignore what sits beside them, which read
  `C1CX` as `C1` and lose the rest without saying so.
- Version 2.12's lettered names are read as the signals they name. It gave the
  L1 and L2 civil signals their own letters and left the digits to the P code,
  so at 2.12 a GPS `LA` is the C/A phase and `L1` the P(Y) one. Every letter was
  read as though it were a band, producing codes like `CAX` that name no signal.
- A version 2 name is only a constellation's where it names a band that
  constellation measures. Version 2 shares one digit space across all of them,
  and the writer would offer GPS a `P5` and Galileo a `C2`, neither of which
  exists.

- `PRN / # OF OBS` is read from the columns the format writes it in. The record
  is `3X,A1,I2,9I6`, so the satellite sits at columns 4 to 6 and the counts
  follow from column 7. Reading the satellite from the first three columns found
  them blank on every conforming file, so the record was dropped and every count
  with it, and the writer put the satellite where the reader had looked for it.
  Files this crate wrote round-tripped; nobody else's did. Reading them now
  fires the `PRN / # OF OBS` quality-control lint on trimmed files, whose
  headers keep counts for a whole day of observations their body no longer has.
  A version 2 file's counts are read too. Version 2 names its observation codes
  once for the whole file and the per-constellation lists are not built until
  the body is read, so taking the count from those lists found none while the
  header was still being read, and every count went. A version 2 record may also
  leave the constellation letter blank, meaning the one the header names, which
  the version 3 satellite reader rejected. Counts are read once the header
  ends, against the observation types it ends with: RINEX 3.05 fixes no order
  between the records, counts declared before their types used to be read
  against no types and lost, and counts read between two declarations for one
  constellation came out shorter or longer than its list. A malformed count is
  still reported where it stands.
- A version 2 observation value is refused when its field cannot write it back:
  wider than `F14.3`, or carrying more than three decimals. Version 3 values
  were already held to this; a version 2 `0.0001` was read and then written
  back as `0.000`.
- A RINEX 2 observation product is written as a RINEX 2 file. It used to be
  re-emitted through the version 3 record writer, so its own output declared
  version 2 in the header while carrying version 3 `>` epoch records: a file
  valid as neither version, which no other reader would take. The writer now
  emits the records the product's version names: the version record naming its
  constellation, `# / TYPES OF OBSERV` continued past nine codes, epoch lines
  carrying their satellite list twelve to a line with the receiver clock offset
  in columns 69 to 80 whatever the satellite count, and observation records five
  values to a line running to the declared count for every satellite. The
  committed 2.11 fixture writes its observation types back byte-identical.
  Records version 3 introduced - `MARKER TYPE`, `SIGNAL STRENGTH UNIT`,
  `SYS / PHASE SHIFT`, `GLONASS COD/PHS/BIS` - and the
  leap-second future count, week and day are written as extension records, the
  way `GLONASS SLOT / FRQ #` already was, because this reader keeps them at
  version 2 and a reader that does not know one skips it. Values are written
  as stored; scale factors are refused, as above.
- **Breaking.** `ObsHeader` carries `rinex2_types`, the version 2
  `# / TYPES OF OBSERV` names as read, empty at version 3. A version 2 file lists
  one set of names for every constellation, and a `PRN / # OF OBS` count for a
  constellation no observation names reads by those names, not by any list in
  `obs_codes`. The writer used to choose names from the lists alone, so a GPS
  count beside a BeiDou observation could be written under a name that reads as
  the same BeiDou code and a different GPS one, and a downgrade to 2.12 could
  write a GLONASS count under a name that no longer counted its code, with no
  change reported. The writer now writes the names a product was read with when
  they still read as every list, and otherwise holds its choice to the lists
  those counts imply; the downgrade lays those implied lists out and reports
  their changes like any other. Observations in version 3 layout in a version 2
  file with no `# / TYPES OF OBSERV` are refused, as observations in version 2
  layout already were.
- Epoch picoseconds are written where RINEX 4.02 puts them, five digits
  (`1X,I5.5`) after the receiver clock offset, whose columns are left blank when
  there is no offset, and read from there. They were written between the
  seconds and the flag, which pushed the flag and satellite count out of their
  columns for every other reader while this one read the line back unchanged,
  and a 4.02 line carrying them where the format does was read with them
  dropped. The long-standing reading of the old placement is kept. Picoseconds
  in a product below 4.02, which introduced them, are refused with
  `RinexObsWriteError::EpochPicosecondsNotInVersion`; a downgrade to version 2
  removes them.
- Picoseconds after the clock offset are read however the epoch line is
  spaced. A tab after them took a correctly laid-out line out of its columns,
  and the whitespace reading then dropped five digits after a clock offset;
  trailing whitespace is now ignored, and in a loosely spaced line five digits
  after a clock offset written with a decimal point or exponent are read as
  picoseconds. A lone five-digit token after the count keeps its reading as a
  clock offset, and a flag and count run together keep theirs. Repair keeps epochs
  at the same second whose picoseconds differ, which it merged as duplicates,
  dropping the second measurement.
- **Breaking.** `ObsHeader` carries `rinex2_system`, the constellation a
  version 2 file's version record names (`None` for a mixed file and at version
  3). With no observations, a version 2 file states that constellation's list,
  so it is part of what the file says: an unused list used to change a
  header-only GLONASS file into a GPS one when written, and a downgrade decided
  the constellation again after adding lists of its own, leaving the GLONASS
  list out of its layout and refusing the conversion. A downgrade now keeps the
  source's constellation and states its list, cleared or not, so a header-only
  GPS C2X list is kept or its change reported when converting to 2.12. The
  version record names the held constellation while every observation is from
  it, and `M (MIXED)` otherwise, so a mixed header stays mixed; a conversion from
  version 3 names the one constellation all observations are from.
- A `GLONASS COD/PHS/BIS` code longer than its three-character field is refused
  when read; it was cut when written and read back as another code.
- A version 2 file is written, and a product downgraded, with names chosen only
  for the lists the file states: one for each constellation an observation or
  count names, and with no observations, the list its version record names. A
  list nothing names no longer makes the writer refuse, nor takes columns, moves
  codes, adds changes or exceeds the type count in a downgrade, and is kept as
  it was; the version record's list of a file with no observations is kept or
  its rename reported even when counts name only another constellation.
- Repair drops empty satellite records before recomputing the observation
  count headers. It counted first, so a file whose satellites held no values
  came out declaring them in `# OF SATELLITES` and `PRN / # OF OBS` with no
  satellite left in its body, and repairing that output changed it again.
- A written version 2 file is compared with its product by what a reader builds
  from it: for each constellation an observation is from, the list its type
  names read as, and with no observations, the list for the constellation its
  version record names. That record names the one observed constellation,
  `M (MIXED)` for several, and with none the product's one list's constellation
  or GPS, so rewriting a file whose observations were all removed no longer
  flips it from `M (MIXED)` to `G`. A product whose lists a caller cleared writes
  its retained type names, header-only included; it was refused though its file
  states everything it holds.
- The writer refuses an epoch flag of 10 or more with
  `RinexObsWriteError::EpochFlagTooWide`, at either version. The reader keeps
  its long-standing reading of such a line, and both versions write the flag in
  one digit, so it used to be written across the satellite count and, at
  version 3, read back as if nothing were wrong.
- A version 2 file's observation records are read by its own
  `# / TYPES OF OBSERV` list. A version 3 `SYS / # / OBS TYPES` record in a
  version 2 header used to replace that list for its constellation, so a file
  whose two records disagreed read its observations into the wrong fields; the
  version 3 record is now reported as not retained, wherever it sits in the
  header. Epoch records in version 3 layout in a version 2 file, which earlier
  releases of this writer produced, are read by the file's own list as well.
- A version 2 name a constellation has no observable under stays as the file
  wrote it, rather than being read as a signal the constellation does have, and
  so does a one-character name in the two-character type field.
  Version 2 names its codes once for every constellation at once, and Galileo
  has no `P1`, BeiDou no `P2`. Reading Galileo's `P1` as its `C1X` put one code
  at two positions, and a writer then had to choose which held the measurement.
  Choosing by name made a valid mixed file grow a column on every rewrite;
  choosing the first position kept a blank and dropped the measurement beside
  it, which on a real mixed file written twice as version 2 lost thousands.
  With the name kept as written each measurement has one position, the header
  keeps its width, and nothing is dropped. The same holds for a second name of
  a code the constellation already has - BeiDou's `C1` and `C2` are both B1I -
  which otherwise grew the header on every rewrite beside a constellation with
  no version 2 names at all. The quality-control lint treats such a name as the
  file's own rather than as a malformed code, so a valid mixed file no longer
  draws an `OBS-H05` finding for every constellation that lacks one of its
  names.
- Version 2.12's lettered names are written as well as read. A 2.12 product
  holding the L1 C/A phase was written under `L1`, which at 2.12 names the P(Y)
  phase, and read back as a different signal. The digits a letter replaced are
  no longer names at 2.12, so a `CA` is written `CA` rather than the `C1` that
  2.12 refuses. GLONASS is given a `P` code on G1 and G2 only, which is where it
  has one. A `PRN / # OF OBS` count
  moves to the column its measurement went to, rather than staying at the
  position it held in its own constellation's list. Two columns naming the same
  code are folded into one wherever no constellation needs both, so two lists
  holding the same signals in a different order no longer double the header.
- CRINEX compresses and expands a RINEX 2 observation file with more than nine
  observation types. Version 2 continues its `# / TYPES OF OBSERV` list on
  records whose count field is blank, and every record was read for a count, so
  the blank one was refused and the whole file with it. A record with a blank
  count now adds its codes and no count; a count that is present and malformed
  is still refused. A continuation record that continues no declared list, one
  naming more codes than its list has left, and a list whose records end short
  of its count are refused, in either version. A later declaration replaces
  the list, as `crx2rnx` applies it.
- CRINEX keeps loss-of-lock and signal-strength flags where `crx2rnx` and
  `rnx2crx` keep them. Expansion carried every satellite's flags from epoch to
  epoch whatever happened in between, so a CRINEX 1 observation flagged, then
  missing, then back unflagged came back with the old flags, and a satellite
  missing from an epoch came back with the flags it had before. Only a
  satellite the previous epoch carried keeps its flags and arcs; a reset or an
  event epoch starts every satellite anew; a blank CRINEX 1 observation's flags
  are blank; a blank field ends its arc; and a blank CRINEX 3 field is written
  with its flags. Compression wrote each satellite's flags as a difference from
  the previous epoch although every epoch it writes is a reset, after which
  `crx2rnx` reads each satellite as new and dropped a repeated flag; it now
  writes them whole. A RINEX 2 blank field carrying flags, which CRINEX 1 cannot
  hold and `rnx2crx` refuses, is refused.
- CRINEX applies the observation type records an event epoch carries. A flag 4
  epoch can declare a new type list, which sets how many fields the epochs after
  it hold; the declaration was copied through and the old widths kept.
- **Breaking.** CRINEX compresses and expands RINEX 4 observation files, and
  carries RINEX 4.02 epoch picoseconds as CRINEX 3.1 does, after the clock
  offset on the clock line. A RINEX 4 file was refused by compression, and a
  CRINEX 3.1 clock line carrying picoseconds by expansion. `ObsEpoch` gains
  `picoseconds`. Compression writes CRINEX 3.1 for RINEX 4.02 and later, as
  `rnx2crx` does. An epoch line with text past its clock offset that is not
  picoseconds its version holds is refused rather than dropped. `rnx2crx`
  leaves an epoch's picoseconds unwritten when they repeat the previous
  epoch's, and writes an epoch with none after one with them the same way, so
  expansion carries an unwritten value forward, as `crx2rnx` does. Compression
  blanks the picoseconds of an epoch with none after one with them, which both
  expansions read as none.
- CRINEX expansion writes a receiver clock offset as `crx2rnx` does, with no
  zero before the decimal point (`.000123000`), where it wrote `0.000123000`.
  `crx2rnx` 4.2.0 misprints a negative offset whose last eight digits are zeros,
  writing `-1.5` as `-1.4`; expansion writes the value. An epoch with no clock
  offset ends the clock's difference arc.
- **Breaking.** A RINEX 2 satellite token with a blank constellation letter
  stays blank through CRINEX, as `rnx2crx` and `crx2rnx` keep it, in
  `SatRecord::sv` too. The header's letter was written in, so a mono-system file
  came back with tokens it did not have.
- The RINEX observation reader refuses a `# / TYPES OF OBSERV` or
  `SYS / # / OBS TYPES` continuation record that continues no declared list. A
  version 2 record with a blank count and codes, before any count or after a
  complete list, and a version 3 record with a blank system field before any
  system, were skipped with their codes.
- The RINEX observation reader takes the records the format lays out in fixed
  columns from those columns, rather than by splitting the line on whitespace.
  Whitespace cannot read a record whose field fills its width, because it then
  touches the field beside it: an epoch of 100 or more satellites writes its
  `I1` flag and `I3` count as `0100`, and a receiver position with a
  -10,000,000 m component writes as `0.0000-10000000.0000`. Both are ordinary
  conforming data and neither parsed. The columns are read only when the line is
  laid out in them, which is decided by whether any content strays into the gaps
  the format reserves; every looser reading still applies beneath that, in turn,
  so a line that is not in the layout keeps the reading it has always had.
  A line that *is* in the layout is now read as the layout describes it, which
  is the intended change and can disagree with what splitting on whitespace made
  of it. `0100` in an epoch's flag and count columns followed by a clock offset
  read as an out-of-range flag of 100 with no satellites, and is now the hundred
  satellites the columns describe; a minute and seconds field run together read
  as one number and are now the two the columns separate. Such a line was
  already being read as something its own layout contradicts. This covers both
  epoch readers, the `APPROX POSITION XYZ` and `ANTENNA: DELTA H/E/N`
  components, and the `TIME OF FIRST OBS` / `TIME OF LAST OBS` fields.
- The reader follows the records a file carries rather than the version its
  header declares. A file declaring version 2 while carrying `>` epoch records
  was rejected, and such files exist: this crate's own writer made them until
  the fix above, and other tools make them too. The distinction is unambiguous,
  since a version 2 epoch line begins with its two-digit year, never with `>`.
- A `GLONASS COD/PHS/BIS` record carrying more entries than one line holds is
  continued on another line instead of being cut off at the sixtieth column, and
  the reader adds each line's entries to the record. A fifth entry was silently
  lost. RINEX gives this record four biases and no continuation, so carrying
  more is an extension of this crate's own; a blank record still means the
  biases are unknown and clears what came before it.

- An epoch of 100 or more satellites now parses. The epoch flag is an `I1`
  field and the satellite count an `I3` field in the next columns, so a
  conforming line reads `0100` with no separator between them, and splitting
  the line on whitespace saw a single token and reported too few fields.
  Multi-constellation files reach that satellite count routinely, so this
  affected ordinary data rather than only generated input. The flag and count
  are separated again, including on the lines whose picosecond or clock-offset
  fields shift the flag away from its usual column.
- The same fixed-column rule now covers the rest of the RINEX observation
  format: the version (`F20.2`), a `GLONASS COD/PHS/BIS` bias (`F8.3`), and an
  epoch record's seconds (`F11.7`), receiver clock offset (`F15.12`) and
  observation values (`F14.3`). Each accepted values the writer could not
  reproduce: a version of `3.999` came back as `4.00`, an epoch second of
  `59.99999999` was written as `60.0000000` and then failed to reparse as a
  civil time at all, a clock offset of `-123.456789012345` overran its columns
  and merged with the satellite count, and an observation value below the
  field's resolution came back as zero. The observation value is now checked for
  precision as well as width, comparing after the scale factor is divided back
  out, because that is the value the next read recovers.
- The six columns RINEX reserves between an epoch's satellite count and its
  receiver clock offset are written again. Without them a full-width negative
  offset abutted the count, and the epoch line no longer read back.
- `APPROX POSITION XYZ` and `ANTENNA: DELTA H/E/N` are read from the columns the
  writer uses, falling back to whitespace for loosely formatted files. The three
  `F14.4` components leave no separator when one fills its field, so a receiver
  position with a -10,000,000 m component was written as
  `0.0000-10000000.0000` and read back as a single token, failing to parse. This
  affected conforming files, not just generated ones.
- RINEX observation header numbers that a fixed-column field cannot re-emit are
  now rejected at parse instead of being written back as a different number.
  `INTERVAL` is an `F10.3` field, the `APPROX POSITION XYZ` and
  `ANTENNA: DELTA H/E/N` components are `F14.4`, and the `TIME OF FIRST OBS` /
  `TIME OF LAST OBS` seconds are `F13.7`, but the parser accepted any finite
  float in those columns. A value below the field's resolution was written as
  zero, which RINEX reads back as the "unknown" zero rather than the original
  value; a value too wide for its columns overran them, so the position and
  antenna-delta lines re-encoded into text that no longer parsed at all, and an
  `INTERVAL` of `1e300` came back as `1000000000`. Scheduled fuzzing found the
  first case through repair idempotence: a header declaring `INTERVAL` at
  `1e-300` on a product with no epochs was kept by the first repair (a positive
  interval is usable), written as `0.000`, and then deleted by the second
  repair because zero is not a usable cadence, so repairing twice did not match
  repairing once. This is the rule the SP3 record fields already applied; it now
  covers the observation header, and both parsers share it.
- Observation repair no longer adopts a cadence the `INTERVAL` header cannot
  record. The inferred cadence is the dominant epoch spacing, so a file whose
  epochs sit a year apart inferred 31,536,000 seconds, which overruns the
  field's ten columns; the repaired file was written with a malformed header
  line. Repair now leaves the header without an interval in that case, and the
  writer omits an interval it cannot express rather than emitting a line that
  reads back as a different number. Repair also replaces a declared interval the
  header cannot record even when it agrees with the inferred cadence to within
  the comparison tolerance, so the repaired product and the repaired text no
  longer disagree about what the file says.

## [2.1.0] - 2026-09-05

### Added

- `Sp3InterpolationOptions`, the policy an SP3 product's node series are read
  with. Its one setting, the gap threshold factor, is the multiple of a
  satellite's nominal spacing above which a consecutive node gap is a coverage
  gap; it was a fixed 1.5 and that remains the default
  (`DEFAULT_GAP_THRESHOLD_FACTOR`, `Sp3InterpolationOptions::DEFAULT`), so
  nothing moves for existing callers. The value is private and set only through
  `new`, which requires finite and greater than 1.0. A factor large enough to
  admit nodes so far from the query that they are no longer distinct at its
  precision makes interpolation return `Error::InvalidInput` instead of a
  position; previously unreachable, since 1.5 never admitted such nodes. `Sp3::with_interpolation_options` sets it
  on a product; `PreciseEphemerisInterpolant::from_sp3`, `StencilExtent::for_sp3`
  and a store written from the product carry it, `PreciseEphemerisSamples` and
  `PreciseEphemerisInterpolant` take it directly, and
  `ContinuityOptions::with_interpolation_options` applies it to the hold-out
  replay. The policy is not SP3 text: it does not survive `to_sp3_string`, it is
  excluded from product equality, and `merge` output carries the default. The
  precise-interpolant store records a non-default factor in previously reserved
  header bytes 48..56; a default-policy artifact is byte-identical to one
  written before, an artifact written before reads back as the default, and a
  factor that is not finite and greater than 1.0 is rejected at open.

### Fixed

- `StencilExtent::for_sp3` reported the SP3 interpolator's reach as five
  product intervals on each side, the centered interior stencil. The
  interpolator selects up to 11 nodes from each satellite's own node series,
  slides that window inward at run edges, serves a query up to one nominal
  spacing outside a run or across a coverage gap, and tolerates gaps up to 1.5
  times the nominal spacing inside a run, so eleven nodes at 0, 600, 1500, ...,
  8700 s span 8,700 s although their nominal spacing is 600 s. The reach is now
  the widest selectable window span over the product's satellites plus one
  nominal spacing, computed with the interpolator's own run and window rules,
  and the influence bounds are measured from the window bounds instead of a
  header-grid snap that could move the upper bound earlier than a selected
  node. The old value erred in the unsafe direction: a window-scoped verdict
  could report Accept for a defect on a node the interpolation used. A wider
  reach means more Refuse decisions near recorded defects; `StencilExtent`
  compares equal for products with equal reach.
- `GnssWeekTow::normalized` could return a time of week outside `[0, 604800)`.
  A TOW a fraction of a nanosecond before the week start borrows a week, and
  the borrow subtraction `tow - (-1 * 604800)` rounds back up to exactly
  604800, because binary64 spacing there is about 1.16e-10; the rounded result
  now carries into the following week. A negative subnormal TOW never borrowed
  at all, because dividing it by the week length underflows to -0.0; it now
  lands on the week start. RINEX 4 CNAV records reached the first case through
  the `top` field: the pair serialized as week `w` with TOW 604800, which
  reparsed as week `w + 1` with TOW 0, so a second encode differed from the
  first. Found by the `rinex_nav_round_trip` fuzz target; the reproducer is in
  the committed corpus. Correcting the pair also corrects the CNAV `dt_op`
  term that subtracts weeks and TOW separately, which for the affected records
  shifts the URA by a few ulp.
- The RINEX navigation writer now normalizes the `top` pair as it will be
  written rather than as stored, and repeats that on the normalized value,
  because normalizing `(9, -1e-8)` yields `(8, 604799.99999999)`, which the
  `D19.12` column writes as a full week again. `GnssWeekTow` has public fields
  that a caller can set outside the normalized range, so the writer cannot rely
  on its input being normalized.

## [2.0.0] - 2026-09-03

### Changed

- **Breaking:** Public input structs are now `#[non_exhaustive]`; external
  construction goes through `Default`/`new` plus field assignment, so adding
  an option to these structs is no longer a breaking change.
- Portable dynamic matrix and matrix-vector products are pinned to nalgebra's
  fixed-order scalar path by a randomized bit-identity test (normal-equation
  orders through 500 and a 2000x200 Jacobian). CI benchmarks every
  application hotpath and portable linear-algebra case against the same-job
  merge base and fails on a regression above 25 percent.
- `nalgebra` 0.33.3 and `simba` 0.9.1 are now exact dependency pins. The
  decomposition algorithms and scalar-dispatch companion participate in the
  crate's bit-exact identity claim, so a semver-compatible update could change
  operation order, convergence thresholds, or covariance bits; any upgrade is
  now a deliberate re-pinning release rather than a side effect of `cargo
  update`.
- Native exact-cache locking now uses the stable standard-library file-locking
  API, removing the unmaintained fs2 dependency while preserving bounded,
  non-blocking retry behavior and error handling.
- **Breaking:** `terrain`, `ionex::tec_grid`, and
  `astro::propagator::dense_output` now return typed error enums from their
  public parsing, interpolation, and dense-output evaluation APIs; the error
  messages remain unchanged.
- `libm` is pinned to exactly 0.2.16. The crate's cross-platform bit-exactness
  is a property of that implementation's rounding, so a semver-compatible
  update of it could change results without any change here. The pin makes
  such a move a deliberate edit with a re-pinning pass, not a side effect of
  `cargo update`.
- OMM parsing and SP3 merging are organized into documented private stages;
  frozen serializer outputs, diagnostics, and numerical results are unchanged.

### Added

- `#![warn(missing_docs)]` is enabled and clean: every public item in the crate
  carries reference documentation stating units, frames, the function that
  produces the value, and the condition under which each error variant is
  returned.
- Reference documentation for every public item in the crate, written from the
  code that produces and consumes each one: units, frames, bit widths and scale
  factors for raw protocol fields, the header record or function each value
  comes from, when an `Option` is `None`, and the condition under which each
  error variant is returned.
- A supply-chain gate (`cargo deny`: advisories, licenses, bans, sources)
  runs in CI, together with an MSRV check at Rust 1.89. One advisory is an
  explicit, documented exception: RUSTSEC-2024-0436 (`paste`, an unmaintained
  compile-time proc-macro reached only through the exact `simba` pin); it is
  revisited when `nalgebra`/`simba` are upgraded.
- Batch APIs now use an optional default-on `parallel` feature. Disabling it
  removes rayon and compiles the same order-preserving batch entry points with
  plain iterators, keeping results bit-identical and retaining the public
  serial variants.
- `JulianDate::new`, `whole`, `fraction`, and `from_unix_microseconds`: named
  construction and accessors for the split Julian date, and a Unix-microsecond
  conversion that shares its floor-and-remainder arithmetic with pass
  prediction (bit-identical, proven by a test over negative and day-boundary
  inputs). The tuple representation is unchanged.

### Fixed

- Bilinear DTED height lookups lost the query coordinate's low bits in the two
  one-degree bands whose tile origin index is -1: the band south of the equator
  and the band west of the prime meridian. The cell offset was formed by
  subtracting the tile origin, which for those tiles is `coordinate + 1`, and
  adding 1 discards everything below one ulp of 1. The interpolation weights
  were then slightly wrong, and a query near a cell boundary could land in the
  neighboring cell. The offset is now measured from whichever tile edge is
  nearer, so the subtraction is exact, and the complement is taken on the exact
  integer ratio. Measured on a synthetic tile at 3600 postings per degree with
  8849 m between adjacent postings, 1,368 of 3,500 probes moved, by up to
  1.1e-9 m; on real SRTM1 tiles the differences are a few times 1e-12 m. Tiles
  with any other origin are bit-identical, and `NEAREST_POSTING` is unchanged
  everywhere.
- Documentation now states the actual single-crate layout: the GNSS layer is
  always present alongside propagation, and the units policy permits bare
  solver-space positions while keeping frame and datum names on georeferenced
  quantities.

## [1.4.1] - 2026-08-31

Supersedes 1.4.0. Solutions are unchanged; a diagnostic is restored.

### Fixed

- A RINEX NAV record probe sliced a line at a fixed byte offset and panicked
  when the header bytes were not a UTF-8 character boundary (found by fuzzing);
  it now inspects bytes and never panics on malformed input.
- A DTED coordinate field ending in a multi-byte character was sliced at an
  invalid boundary and panicked (found by fuzzing); it now returns a typed
  error.
- The core trust-region backend no longer overrides the solver's dot-product
  and matrix-vector reductions. Those fallbacks were already portable
  fixed-order arithmetic; overriding them in 1.4.0 summed the terminal
  gradient in a different order, which left every converged solution
  bit-identical but moved the reported first-order optimality (a
  cancellation-dominated number) by an order of magnitude on some fits, and
  could change the evaluation counts of a fit. 1.4.1 reports the same
  optimality and counts as 1.3.3 for fits whose path is otherwise unchanged.

## [1.4.0] - 2026-08-31

Results are bit-identical across x86_64 and arm64 targets. Relative to 1.3.3
the frozen outputs of iterative fits move in their last bits, and SVD-derived
covariance and geometry diagnostics now agree with the values 1.3.3 produced
on x86_64/glibc; position and clock results of the bundled static and SPP
fixtures are unchanged.

### Changed

- All core transcendental evaluation now delegates to the portable `libm`
  kernels, including the trust-region solve paths and numerical weighting,
  so solved results do not depend on the platform C math library.
- Core nalgebra decompositions and dynamic matrix products now run through a
  transparent portable binary64 scalar. This bypasses architecture-selected
  SIMD matrix-product kernels while retaining binary64 arithmetic and making
  SVD-derived covariance and geometry diagnostics bit-identical across targets.
- Core trust-region solves now inject their own portable numerical backend for
  SVD, powers, dot products, and matrix-vector products; the general-purpose
  trust-region crate's default backend and published parity behavior are
  unchanged.
- Robust-loss evaluation in core-driven trust-region solves (Cauchy, Arctan)
  uses portable `log1p` and `atan` through the new defaulted hooks in
  `trust-region-least-squares` 0.11.0.
- Fused multiply-add in the SP3 interpolant, geoid, frame, and vector helpers
  goes through `libm::fma` instead of the platform C library's `fma`; no
  result changed on any tested platform, the dependency did.
- A workspace lint (`clippy.toml` `disallowed-methods`) rejects any new
  platform-libm transcendental or `mul_add` call, and a guard test keeps
  production decompositions on the portable scalar.

## [1.3.3] - 2026-08-30

### Fixed

- The Moon's geocentric distance in the analytic Sun/Moon series lost its
  parallax sine in 1.3.2 (`a / parallax` instead of `a / sin(parallax)`),
  moving the Moon by about 17 km. The solid-earth tide it feeds was off by
  roughly 140 mm. 1.3.2 is superseded; every downstream interface skips it.
  A bit-pinned Moon regression now guards the series, since the DE440 golden
  tolerates the model's own ~1% and cannot see a slip of this size.

## [1.3.2] - 2026-08-30

### Fixed

- `parse_archive_listing` deduplicated by scanning every object parsed so far
  for each incoming row, which is quadratic. On AIUB's whole-tree CSV
  (~426k rows) the parse took 154 s; it now indexes each path's position and
  takes 0.23 s. Listing order and the `observed_at` backfill are unchanged.

### Changed

- Transcendental math (sin, cos, tan, atan2, asin, acos, exp, log, pow) now
  goes through portable Rust kernels rather than the platform C math library,
  so results are bit-identical across x86_64 and arm64. The full test suite
  now runs on both architectures in CI and passes bit-for-bit on each. The
  owned trust-region solver's complete subproblem assembly, not only its
  factorization, uses fixed-order scalar arithmetic.

## [1.3.1] - 2026-08-29

### Changed

- Coordination release keeping the shared release number across the language
  interfaces. Ships the Go interface relicense from Apache-2.0 to MIT
  (matching the engine and every other language interface). No numerical,
  algorithmic, or API changes in the engine.

## [1.3.0] - 2026-08-29

### Changed

- Coordination release keeping the shared release number across the language
  interfaces, which now include a Go interface. No numerical, algorithmic, or
  API changes in the engine.

## [1.2.0] - 2026-08-28

### Added

- `rinex_band_frequency_hz_classified` reports
  `Error::MissingGlonassChannel` when GLONASS G1/G2 needs an FDMA channel,
  while existing lookup behavior and frequency values remain unchanged.

### Fixed

- Lenient RINEX 4 NAV parsing now decodes GPS/QZSS CNAV-family EPH frames
  instead of skipping them.
- RTKLIB SBAS parsing now preserves the `Framed250` wire form for valid
  32-byte blocks without changing the parsed bytes.

## [1.1.1] - 2026-08-26

### Changed

- Coordination release restoring the shared release number across the language
  interfaces after the Elixir 1.1.1 patch. No numerical, algorithmic, or API
  changes.

## [1.1.0] - 2026-08-24

### Added

- `locate_source_with` and `SourceLocateConfig`, a `#[non_exhaustive]`
  configuration wrapping `SourceLocateOptions`, whose `include_influence`
  can skip the one-full-re-solve-per-sensor leave-one-out diagnostics and
  return an empty influence vector. `locate_source` is unchanged and
  equivalent to `include_influence = true`.
- `closed_form_initial_guess` names the source-localization seed for its actual
  Schau-Robinson spherical-intersection method. `chan_ho_initial_guess` remains
  as a deprecated compatibility wrapper.

### Changed

- `SourceLocateOptions` keeps its 1.0 shape; settings added from here on live
  on `SourceLocateConfig` so they stay additive.
- Source-solution rank, condition number, covariance, and GDOP now come from one
  thin SVD of the final Jacobian. The covariance is assembled as
  `V * diag(1 / sigma_i^2) * V^T` over retained singular values instead of a
  separate Cholesky inverse of the normal matrix.
- Sensor influence `score` is exactly the larger absolute full/leave-one-out
  ToA residual divided by `timing_sigma_s`. Robust downweighting remains
  available separately in `loss_weight`.
- TDOA origin time uses one robust-loss reweighting refinement after the
  position solve when a non-linear loss is selected. Linear loss retains the
  original arithmetic-mean path exactly.
- Source-localization documentation now specifies the ToA/TDOA models, state,
  sensor minima, seed method, covariance/CRLB interpretation, influence cost,
  solver termination codes, and fallible API errors.

### Fixed

- Closed-form quadratic degeneracy and discriminant checks are relative to the
  coefficient magnitudes. The ToA seed no longer rejects an otherwise finite
  candidate with an arbitrary absolute-distance cutoff.
- Empty source-localization sensor input now reports `InvalidInput` for
  `sensors` instead of a dimension-assuming `TooFewSensors { needed: 3 }`.

## [1.0.1] - 2026-08-22

### Changed

- `trust-region-least-squares` updated to 0.10.0: the injected backend contract
  is now the `HostNumerics` seam (SVD, BLAS reductions, and NumPy power
  dispatch in one fail-closed contract), and the host backend reproduces
  NumPy's stride-0 scalar-exponent power fast paths bit-for-bit. sidereon-core
  drives the solver through its data/model entry points and is unaffected at
  its own API surface.

## [1.0.0] - 2026-08-21

Sidereon 1.0.0. The public API carries a stability commitment from here:
additions arrive without breaking existing callers (MergeOptions and its
non-exhaustive construction pattern are the template), and anything that
must break waits for 2.0.0.

### Added

- Window-scoped continuity verdicts: `EpochWindow`, `StencilExtent`
  (derived from the interpolator's sliding-window order and the product's
  epoch interval, never caller-supplied), `defects_influencing` on
  `ContinuityReport` and `MergeReport`, and accept/refuse verdict helpers
  that name the influencing defects either way. A consumer evaluating a
  bounded span no longer refuses a product for a seam its stencil cannot
  reach, and cannot silently accept one whose stencil reaches it. The
  `inspect` CLI gains `--window FROM THROUGH`.
- `next_issue_due`: a network-free answer, over the same catalog the
  publication-status query uses, for when the next issue of a cataloged
  product line is nominally due, naming the ultra lines' observed and
  predicted halves. Schedules cited to the published IGS product
  descriptions in committed provenance; boundary behavior pinned across
  UTC midnight and a GPS week rollover. The scoreboard prints the next
  due issue beside the current lag.
- Oracle version pinning documented in `docs/oracle-version-pinning.md`
  with measured (not transcribed) cross-version deltas for the SciPy and
  NumPy reference stack, and a one-command reproduction from pinned
  environments.

(0.40.0's exact-cache single-flight coalescing and non-exhaustive
`MergeOptions` ship to every interface with this release.)

## [0.40.0] - 2026-08-21

### Added

- `ExactProductCache::open_single_flight`: concurrent requesters for one
  product identity coalesce onto a single download. A waiter observes the
  owner's in-flight marker and blocks, bounded, on the committed entry
  instead of re-downloading - the answer to the alias-prone ultra-target
  pairs that previously had to stay serial. Ownership is a random
  128-bit token (PID and wall clocks are diagnostic only, so containers
  sharing a cache directory cannot be confused); liveness is append-only
  heartbeat growth judged on each waiter's own monotonic clock; takeover
  re-verifies the marker snapshot under the existing transition lock and
  claims by exclusive creation; a slow live owner yields a bounded
  `SingleFlightTimeout`, never a second download. The sidecar is
  schema-v3-compatible - commit encoding and `current.json` are
  byte-unchanged - and the mixed-version matrix is documented in
  `docs/exact-cache-single-flight.md`. Nine new failpoint boundaries
  carry process-kill tests; real-process integration tests cover
  waiter-observes-commit-without-downloading, SIGKILLed-owner takeover
  with exactly one commit, and live-owner timeout.

### Changed

- **Breaking**: `MergeOptions` is `#[non_exhaustive]`. It gains a field
  whenever the merge learns a new policy - 0.37.0 alone added two - and
  each addition was source-breaking for every downstream exhaustive
  literal. Construct by mutating `MergeOptions::default()`; future
  options then arrive without breakage.

## [0.39.1] - 2026-08-11

### Fixed

- DTED terrain lookups now compute the grid cell and intra-cell fraction
  in exact integer arithmetic. The 1-arc-second scaling is a roughly
  65-bit product, so the binary64 multiply rounded away the low fraction
  bits - up to 4096 ULP at representative CONUS coordinates - and, at a
  posting boundary, could round a coordinate strictly below a posting
  onto the exact integer, flipping the lookup into the next cell's
  stencil with fraction 0.0. The offset's integer significand is now
  multiplied by postings-per-degree before the power-of-two division,
  with Euclidean flooring for negative offsets and correctly rounded
  dyadic-to-binary64 conversion. All three lookup paths (bilinear DTED,
  bilinear mmap store, nearest-posting) share the one helper; the
  nearest-posting ties-to-even policy is unchanged, now computed on the
  exact remainder. Dyadic-exact coordinates are byte-identical before
  and after.

## [0.39.0] - 2026-08-10

### Added

- Attested opens for both mapped artifact readers:
  `MmapTerrain::from_path_attested` / `from_vec_attested` and
  `MmapPreciseEphemerisInterpolant::from_path_attested` /
  `from_vec_attested`, taking a caller-attested content checksum in place
  of the O(payload) hash pass the verified constructors perform. 0.38
  mapped the file but still hashed every payload byte at open (~90 s cold
  / ~47 s warm on a ~34 GB store, measured downstream); a caller who
  already holds a trustworthy measurement - fs-verity, a signed manifest,
  a content-addressed store - can now hand it over instead.

  The handle carries its digest provenance (`DigestProvenance::Verified`
  vs `Attested`) everywhere the digest appears, so an attested handle can
  never masquerade as a verified one. `checksum64()` on an attested
  handle returns the claim without hashing. `verify()` escalates to the
  full hash pass on demand and flips provenance on success. Everything
  O(header) and O(index) stays unconditional; the interpolant's attested
  open cross-checks the claim against the header's declared checksum in
  O(8) and fails closed with `AttestedChecksumMismatch` - a wrong digest
  for the file is caught without hashing a byte. The terrain header
  carries no file-level checksum, so its claim is recorded as-is and
  checked only by `verify()`.

## [0.38.0] - 2026-08-09

### Added

- `mmap` feature (off by default). With it enabled, `MmapTerrain::from_path`
  and `MmapPreciseEphemerisInterpolant::from_path` memory-map the file
  read-only and the reader owns the mapping, instead of reading the whole
  artifact into process memory. The entry point is unchanged, so every
  existing caller benefits without migrating to a new constructor.

  The copy avoided is the smaller half. A mapping is demand-paged, so a
  reader that queries a geographically local region faults in the pages
  covering those tiles and never touches the rest; construction parses only
  the header, datum tag, and index. That is the difference between opening a
  30+ GB terrain store and being unable to start. Measured on the committed
  fixture: a mapped open allocates ~1 KB regardless of artifact size, where
  the copying open allocates the artifact.

  Neither reader becomes self-referential and no `unsafe` appears at any
  interface boundary. Bytes live in a new `ArtifactBytes` enum
  (`Borrowed` / `Owned` / `Mapped`) and every lookup derives its span on
  demand. The interpolant's mapped parse uses offset-backed arrays rather
  than the borrowed `&[f64]` arrays its borrowed path uses, so the promotion
  to an owning reader is expressible in safe code.

- `MmapTerrain::is_memory_mapped` and
  `MmapPreciseEphemerisInterpolant::is_memory_mapped`, so a caller or a test
  can assert that a path open actually mapped rather than read. A change that
  quietly relocated the copy would otherwise be indistinguishable from a fix.

## [0.37.0] - 2026-08-09

### Added

- `check_continuity` attests that a precise-ephemeris sample series is
  physically continuous, or reports each violation with the epochs, the
  interval, and the magnitude that exceeded its bound. Two checks with
  different jobs: a speed gate whose bound is a true physical upper bound
  for the orbit class (`sqrt(mu/a_min) + omega_e*r_max`), so it cannot
  false-positive and catches gross corruption; and a hold-out
  interpolation residual evaluated through the product's own Lagrange
  substrate, which supplies the sensitivity. On a real GFZ ultra product
  earth-fixed chord speeds run 2757-3187 m/s against a ~6 km/s class
  bound, leaving hundreds of kilometres of displacement undetectable per
  epoch pair, so a speed gate alone cannot see a metre-scale splice; the
  residual check resolves a 5 m splice against a 1 m tolerance. Ordering
  is the library's responsibility - input is sorted internally, so a
  shuffled sequence and a sorted one produce identical reports - and
  after duplicate epochs are split out as their own defect class, a zero
  or negative interval is unrepresentable in the comparison path.
- `MergeOptions::provenance` records per-epoch merge provenance as the
  merge decides: which contributor supplied each accepted cell, where
  selection changed and why, and what each contributor covered.
  `Summary` mode is bounded by the number of selection changes; `Full`
  adds one entry per accepted cell. `MergeReport::provenance` is an
  `Option` so "not requested" stays distinguishable from "one
  contributor". `CellSelection` records a combined value as combined
  rather than nominating a supplier: under `Mean` or `Median` the written
  value is a combination of the members, so no single contributor
  supplied it.
- `MergeOptions::verify_continuity` runs the continuity check over the
  merged product as a post-condition and attributes each violation to the
  contributors on both sides, distinguishing a splice across a
  contributor change from a discontinuity inside one contributor's arc.
  It reports without refusing: the merge still returns the product.

### Changed

- `MergeOptions` gains the `provenance` and `verify_continuity` fields.
  This is source-breaking for exhaustive struct literals; construction
  sites using `..MergeOptions::default()` are unaffected. Neither option
  changes the merged product - the SP3 output is byte-identical whether
  or not they are enabled, pinned by test.

## [0.36.3] - 2026-08-04

### Fixed

- `parse_archive_listing` no longer rejects an AIUB whole-tree CSV listing
  over a path containing spaces. `;` is the field delimiter, so a space is
  legal path content, and the live 426k-row listing carries unrelated
  objects (conference PDFs, tarballs) with spaces in their names; one such
  row rejected the entire listing, so `publication_status` for every CODE
  line followed the redirect and then died at the parser. The four-field
  structure remains the malformed-row signal; closed dialect detection is
  unchanged. Found by downstream 0.36.1 verification; the recorded fixture
  now ends with the verbatim offending row, and the full live listing
  (425,132 objects) is the reproduction.

## [0.36.2] - 2026-08-04

### Changed

- Version-alignment release; no engine changes. The Python interface's
  0.36.2 adds anonymous-FTP transport for the `wum_nrt` line (parity with
  Elixir 0.36.1) and its release gate enforces exact engine-version
  lockstep.

## [0.36.1] - 2026-08-04

### Changed

- Version-alignment release; no engine changes. The Python and WASM
  interfaces enforce exact version lockstep with the engine crates, and
  their 0.36.1 patch (accepting the `WUM` publisher and `near_real_time`
  solution-class tokens in caller-built identities) requires matching
  engine versions on the registry.

## [0.36.0] - 2026-08-04

### Added

- Added an opt-in cross-line candidate walk for CODE's predicted ionosphere:
  `predicted_ionex_line_candidates` enumerates the `P1` and `P2` artifacts for
  one map date (both lines publish the same official filename for a map date,
  but the two-day line is produced a day earlier, so `P2` is routinely
  published while `P1` is still absent when CODE runs behind). Candidates
  never substitute a neighboring date's map, each keeps its own exact
  identity and cache path, and `resolve_first_published` preserves the line
  actually served in provenance. Single-line requests keep their fail-closed
  behavior.
- Added a publication-status API: `parse_archive_listing` (Apache and XHTML
  autoindexes, AIUB's whole-tree CSV, FTP `LIST` output - each verified live
  on 2026-08-04 and recorded as fixtures), `newest_published_product`,
  `published_issue_age_minutes`, and the bounded `publication_listing_urls`
  (current week directory plus previous, or one whole-tree listing). The
  scoreboard's one-call `publication_status` query reports the newest
  published issue and its lag behind nominal without fetching product bytes,
  and reports a transport failure as `Unreachable` rather than answering
  from an older directory - "nothing published" and "archive did not answer"
  are distinct outcomes.
- Added Wuhan University's hourly MGEX near-real-time orbit line
  (`wum_nrt`, `WUM0MGXNRT`, 02D span at 05M over anonymous FTP), verified
  against the live archive: the series begins 2024-07-03 (GPS week 2321) and
  the previously published `WUM0MGXULA` hourly line ended around GPS week
  2230 with a publication gap between; pre-NRT dates are refused. The line
  is not projected onto CDDIS (no exact mapping is cataloged). `ArchiveProtocol`
  gains `Ftp`, `SolutionClass` gains `NearRealTime`, and `ProductPublisher`
  gains `Whu`.
- The IGS combined ultra (`IGS0OPSULT`) and the Wuhan NRT line participate
  in the multi-center SP3 merge-consensus path behind their catalog entries,
  with exact-validation agency pins (`IGS`, `WHU`) and a four-center
  merge-input identity test alongside ESA/GFZ.
- Documented the case for broadcast ephemerides as the acquisition
  resilience floor (`docs/broadcast-ephemeris-resilience-floor.md`), as a
  design issue without implementation.

## [0.35.1] - 2026-08-01

### Fixed

- The RTK double-difference row builder now rejects a rover position that
  overflows to infinity instead of panicking. The rover is formed as
  `base + baseline_m`, and the boundary check validated each operand
  separately, so two individually finite inputs near `f64::MAX` summed to an
  infinite position and tripped a debug assertion inside the internal `add3`
  primitive. The sum is now built with the checked helper and surfaces as a
  typed `InvalidInput { field: "rtk.rover_pos", kind: NonFinite }` through all
  three RTK paths. Physically realizable baselines are unaffected. Added the
  scheduled-fuzz crash artifact (run 30695232523) as a committed corpus seed.
- SP3 now rejects an epoch-record (`*`) seconds value its own field cannot
  re-emit. The writer renders the epoch instant through an `F11.8` field, so
  seconds carrying more precision silently shifted the epoch on re-encode
  (`0.0000009999` came back as `0.00000100`, moving the instant by ~0.1 ns).
  This completes the fixed-column re-emission rule already applied to record
  values and the header line-2 fields. Conforming products are unaffected:
  their epoch seconds already round-trip through the field exactly.
- The RTK row builder now validates a supplied receiver-antenna calibration.
  A non-finite `pco_neu_m` reached the PCO/NEU projection and tripped a debug
  assertion inside the vector primitives; it is now rejected by field as
  `InvalidInput { field: "rtk.receiver_antenna.{base,rover}.pco_neu_m" }`. An
  offset that is finite but large enough to overflow when its three basis
  components are summed is reported as `ReceiverAntenna(InvalidGeometry)` by
  the projection itself. Published antenna calibrations are unaffected: real
  PCOs are centimetre-scale.

### Testing

- `sp3_round_trip` now compares the product's public content - header, epoch
  instants, comments, and every epoch's satellite states - instead of asserting
  whole-struct equality against the pre-normalization product. `to_sp3_string`
  is a normalizing writer, and `Sp3` retains raw acquisition-validation
  provenance describing the *input* text, so `parse(write(x)) == x` was false by
  construction for malformed or sparse inputs and reported those as crashes.
  The content comparison still catches a writer that drops or mangles data.
- `fuzz_rtk` now exercises the receiver-antenna path. It previously passed
  `None` for the corrections at all three RTK entry points, so PCO/PCV
  projection was never fuzzed; the harness now supplies arbitrary base/rover
  calibrations on roughly half of inputs and keeps the `None` path covered.
- RINEX observation headers now reject a code list the fixed-column format
  cannot carry, instead of parsing into a product that cannot be serialized.
  Observation descriptors are `A3` fields in `SYS / # / OBS TYPES`,
  `# / TYPES OF OBSERV`, `SYS / SCALE FACTOR`, and `SYS / PHASE SHIFT`
  (RINEX 2.11 section 5.1, RINEX 3.05/4.02 section 5.1), and the
  `SYS / # / OBS TYPES` count is an `I3` field. A wider descriptor or count is
  now a typed `Error::Parse` at the record that carries it. Previously such a
  header re-emitted a record that overran its 60-column content area, was
  truncated, and re-parsed with fewer codes than its own count declared, so
  `repair -> to_rinex_string -> parse` failed on input the parser had accepted.
- Added the exact 632-byte scheduled-fuzz crash artifact (run 30197879510) as a
  core regression and a committed fuzz-corpus seed.
- A `SYS / PHASE SHIFT` correction far from unity is now written in exponent
  form. Rust's `Display` never switches to an exponent, so a value such as
  `1e-300` rendered as 302 columns of plain decimal: the record was truncated
  into the content area, the correction collapsed to zero, and the satellite
  list disappeared. Corrections that fit the record's `F8.5` field keep their
  existing plain-decimal spelling byte for byte.
- A `SYS / PHASE SHIFT` satellite list that cannot be re-emitted inside the
  60-column content area is now a typed `Error::Parse`. Single-digit PRNs are
  read from two-column tokens (`G1`) but written into `1X,A3` fields, so a
  readable record was not always a writable one.

- SP3 now rejects a value its own fixed-column field cannot re-emit unchanged.
  Record positions, velocities, clocks, and clock rates are `F14.6` fields, and
  the header line-2 (`##`) seconds-of-week, epoch interval, and MJD fraction are
  `F15.8`, `F14.8`, and 13-decimal fields (SP3-c section 3, SP3-d Hilla 2016).
  A value carrying more precision than its field expresses, or one too wide for
  its columns, is now a typed `Error::Parse`. Previously such a value fit the
  columns but not the format, so `parse -> to_sp3_string -> parse` silently
  changed it (`36.019431257` km came back as `36.019431`). Conforming files are
  unaffected: their values already round-trip through their own fields exactly.

### Documentation

- Pinned the SP3 serialization contract for unrepresentable satellites:
  `Sp3::skipped_records` counts entries the input text carried but the product
  cannot represent (an extended GLONASS slot such as `R28` beyond the engine's
  PRN cap). They are deliberately dropped rather than aborting the parse, so
  nothing of them reaches the writer and a re-encoded product always re-parses
  with no skips. The `sp3_round_trip` fuzz target asserted that the two counts
  matched, which no correct implementation can satisfy for such a file; it now
  asserts the re-encode reports zero skips - stricter, since the writer must
  never emit a record the parser cannot represent - while still comparing every
  other field. Added a core regression and two committed fuzz-corpus seeds from
  scheduled run 30262991024. No parser, writer, or numerical behavior changed.

### Compatibility

- Parser compatibility patch. Conforming RINEX 2/3/4 observation files are
  unaffected: their descriptors already fit the `A3` code fields and their
  per-system counts the `I3` field. No public API, numerical kernel, or output
  formatting changed.

## [0.35.0] - 2026-07-24

### Fixed

- Observation QC no longer panics when a successfully parsed RINEX OBS product
  carries `INTERVAL = 0`. RINEX 2.11 section 5.3 and RINEX 3.05/4.02 section
  6.5 permit zero, a blank field, or an omitted optional record when metadata is
  unknown. Blank `INTERVAL` fields are now parsed as absent; zero is retained
  as unavailable metadata and reported as informational `OBS-H19`.
- An unavailable source interval is never used as cadence. QC instead labels a
  cadence inferred from the actual epoch grid as `Inferred`, or reports
  `Unresolved` and skips interval-dependent gap calculations. Negative or
  caller-constructed non-finite source intervals produce error `OBS-H20`.
  Explicit zero, negative, or non-finite caller overrides continue to return
  `InvalidInterval`, and interval repair remains opt-in.
- Gap accounting now saturates instead of overflowing for an extremely small
  positive caller interval, ignores non-finite public-structure epoch deltas,
  and never rounds a sub-millisecond inferred interval down to zero.
- Added the exact 583-byte scheduled-fuzz crash artifact, its human-readable
  reduction, core and CLI regressions, and a committed fuzz-corpus seed. The
  scheduled workflow timeout is now 90 minutes: its 31 two-minute target
  budgets alone require 62 minutes before runner setup and compilation.

### Compatibility

- Parser and QC compatibility patch. `Finding` gains the additive,
  non-exhaustive `ObsIntervalUnavailable` and `ObsInvalidInterval` variants.
  Existing positive source intervals and explicit positive overrides behave as
  before. Solver, orbit, propagation, positioning, frame, timing, and other
  numerical kernels are unchanged.

## [0.34.0] - 2026-07-21

### Added

- Added `supported_samples`, a date- and issue-aware catalog query for the
  officially evidenced cadences of one product. The same gate now rejects a
  syntactically plausible but unpublished cadence before filename, URL,
  identity, or cache-key derivation.
- Added the product- and issue-aware `sp3_content_start_convention` catalog
  query and `Sp3ContentStartConvention`. The result states whether an exact
  SP3 product starts at its filename epoch or one day earlier and exposes the
  corresponding whole-second offset. Invalid issues and issues on product
  lines that do not publish them are rejected.

### Fixed

- Exact SP3 validation now recognizes the terminal record as a complete logical
  record: `EOF` in columns 1-3 followed only by ASCII-space padding, bounded by
  Sidereon's 80-column interoperability policy. This accepts both bare records
  and the padded records published by the audited ESA and GFZ product lines,
  with LF, CRLF, or no final line separator. The previous whole-line equality
  check falsely reported `MissingEof` for those valid public products.
- Malformed EOF-like records now report `MalformedEofRecord` rather than being
  misdiagnosed as absent. Missing markers, `EOFX`, tab padding, padding beyond
  the policy width, leading whitespace, lone-CR framing, premature markers, and
  nonblank data after a valid marker still fail closed. Empty and ASCII-space-
  only records after the marker remain an explicitly documented Sidereon
  tolerance.
- Exact requests derived from historical GFZ ultra-rapid identities now
  distinguish the epoch encoded by the official filename from the product's
  first content epoch. Products through 2022-09-06 require the archive-observed
  one-day offset; the non-monotonic 2022-09-07/08 transition is cataloged per
  issue, and products from 2022-09-09 remain aligned. Declared-start, line-2,
  first-epoch, cadence, grid, and span checks remain strict, and callers cannot
  override the cataloged offset.
- Ultra-rapid location candidates now contain only dated span/cadence variants
  evidenced for the exact center, date, and issue. This removes speculative
  cross-cadence and alternate-span URLs; the documented GFZ `0000` overlap on
  2021-05-15 remains. CODE's moving latest-product snapshot is excluded because
  it is not the dated one-day exact product. All caller-built product
  identities now require the cataloged span, not merely a
  syntactically valid span embedded in a matching filename.
- Hardened auxiliary gzip ingestion for the Rust Bias-SINEX/CODE DCB path
  loaders and the validation scoreboard. They now decode every RFC 1952 member,
  accept optional header fields up to the archive limit, validate FHCRC plus
  every member CRC32/ISIZE and trailer, and reject truncation or trailing data.
  Local loaders enforce explicit 64 MiB archive and 500 MiB product limits.
  Scoreboard downloads use one authoritative bounded GET: final 404/410 remains
  ordinary publication absence, while transport and 5xx retries start with a
  fresh process and buffer so partial attempts cannot contaminate a success.
  Its curl status is carried in a dedicated terminal frame, and publication
  absence is authorized only by curl's HTTP-failure exit plus 404/410; a
  truncated transfer whose partial body ends in those digits remains a
  transport failure.

### Compatibility

- Parser, catalog, and transport compatibility only. No orbit, propagation,
  merge, positioning, solver, frame, timing, or other numerical calculation
  changed.
  All language interfaces inherit the same core behavior; each interface
  carries its own exact-parser regression, and the acquisition-owning
  interfaces also test the complete acquisition path. The GFZ correction
  changes only which cataloged start instant exact validation requires.
- This is a minor release because the new catalog query and enum are public and
  because exact validation now applies newly cataloged historical GFZ
  semantics. `ultra_sp3_locations` can return fewer candidates because
  unsupported alternate spans/cadences and CODE's non-exact moving snapshot
  are no longer represented as dated products. Caller-built identities with a
  noncatalog span now fail validation. Existing SP3 parsing and all numerical
  APIs remain compatible.
- Gzip changes affect transport integrity and resource limits only; no product
  parser, orbit, positioning, merge, or other numerical calculation changed.

## [0.33.1] - 2026-07-20

### Fixed

- Included the repository MIT license, the intact IERS Conventions derived-work
  notice, and the applicable ERFA and RTKLIB notices in the crates.io source
  package. The tide source now points directly to the packaged IERS terms.
- Renamed the private Rust translations of the IERS/SOFA companion routines and
  added the license-required statement that the derived work is not distributed
  or endorsed by the IERS Conventions Center.

### Compatibility

- Packaging, licensing documentation, and private source identifiers only;
  public APIs and numerical behavior are unchanged from 0.33.0.

## [0.33.0] - 2026-07-20

### Added

- Added IGS combined final-SP3 catalog support with date-aware official names:
  legacy `igs<week><day>.sp3` identities from GPS week 0730 through 2237 and
  `IGS0OPSFIN_<epoch>_01D_15M_ORB.SP3` identities from week 2238 onward.
  Historical CDDIS locations use `.Z`; current CDDIS and direct-BKG locations
  use `.gz`.
- Added `product_solution_class` so callers can distinguish IGS final SP3 from
  IGS broadcast navigation without changing the legacy center-only query.
- Added `default_sample_for_date` for product lines whose published sampling
  interval changed over time. GFZ rapid SP3 resolves to `15M` through 2021 day
  137 and `05M` from day 138.
- Added verified series floors for ESA final SP3/clock, GFZ rapid SP3/clock,
  and IGS, CODE, ESA, and GFZ ultra-rapid SP3 products. Ultra issue lookback
  stops at the applicable floor instead of emitting a previous-day identity.
- Made omitted ultra-rapid SP3 cadence issue-aware. ESA uses `15M` through the
  2025-02-02 0600 issue and `05M` from 1200; GFZ uses `15M` through 2021-05-15
  and `05M` from 2021-05-16. Candidate order follows the issue-era default.
- Added `ExactSp3Request`, `parse_exact_sp3`, and `validate_exact_sp3`. Exact
  validation binds the line-1 start/count, line-2 GPS-week/seconds-of-week/MJD
  start metadata, mandatory header/EOF records, producing agency, complete
  per-epoch satellite record sequences, finite positive cadence, parsed regular
  epoch grid, requested cadence, requested span, and optional format revision.
  It accepts both the half-open and inclusive regular-grid representations of
  an exact span.

### Fixed

- RINEX observation repair now canonicalizes malformed non-ASCII and control
  characters before fixed-column header parsing while preserving byte offsets,
  so repaired output remains printable, parseable, and byte-idempotent.
- CODE rapid and final catalog entries now use AIUB's current HTTPS download
  service with product-specific routes for MGEX final SP3/clock, operational
  final IONEX, and rapid IONEX; the already-correct ultra-rapid SP3 route is
  preserved. Historical `cod` requests are rejected until their distinct
  short-name identities are modeled instead of being assigned current long
  filenames.
- Caller-built identities for unsupported center/product combinations now fail
  before URL derivation or acquisition.
- IGS combined final-SP3 requests now reject dates before the official start at
  GPS week 0730 (1994-01-02), and legacy CDDIS product paths zero-pad the GPS
  week directory to four digits.
- CDDIS location derivation rejects pre-week-2238 long-name SP3 and IONEX
  identities while retaining the verified IGS final short-name `.Z` series.
  It also refuses to substitute a different CDDIS product for ESA's exact
  `ESA0MGNFIN` final-SP3 identity.
- Corrected the current GFZ rapid-SP3 catalog default to `05M`. Date-derived
  requests preserve the historical `15M` default through 2021 day 137 and use
  the published `05M` convention from day 138, including current products.
- Exact-SP3 candidate selection now advances only after ordinary publication
  absence. Parse, digest, identity, cadence, grid, and span failures remain
  terminal and preserve the first integrity error rather than accepting a
  later candidate. Candidate product codes and SP3 producing-agency fields are
  bound to the selected public product family.
- SP3 serialization now pads the mandatory header comment section to four
  records without adding semantic comments. Exact validation also enforces the
  line-3 satellite count, per-epoch record order, and P/V pairing required for
  velocity products.

### Compatibility

- Existing IGS broadcast-navigation derivation and
  `AnalysisCenter::solution_class()` are unchanged. The product-aware query and
  exact-SP3 validator and date-aware default-sample query are additive;
  `ArchiveCompression::UnixCompress` adds a public enum variant, and the
  catalog and scoreboard errors add typed public variants. The legacy
  date-free `default_sample` now reports GFZ's current `05M` rapid-SP3 cadence;
  dated default derivation remains `15M` for historical products through 2021
  day 137. For issue-based products, the date-only default represents the 0000
  issue while product construction uses the actual issue. Invalid identities,
  pre-series dates, unsupported combinations, unmodeled historical CODE
  products, and integrity-invalid exact SP3 content now fail earlier.
  Serialized SP3 text with fewer than four semantic comments gains blank
  mandatory comment records; blank structural padding is no longer surfaced as
  semantic text in `Sp3::comments`.
  These public additions and stricter semantics require a minor `0.33.0`
  release rather than a patch.

## [0.32.0] - 2026-07-18

### Added

- Added deterministic `parse_navcen_at` and `merge_navcen_at` APIs for
  evaluating NAVCEN operational usability at an explicit UTC instant. Returned
  assessments preserve NANU type, subject, raw Outage Start text, the evaluation
  instant, and parsed/unparseable/not-applicable timing provenance.

### Fixed

- Active bounded forecast NANUs now affect the time-aware path only during
  their validated half-open UTC interval. Future forecasts no longer disable a
  satellite early, completed temporary outages no longer remain active, and an
  incomplete interval remains usable with explicit ambiguity provenance.

### Compatibility

- Existing `parse_navcen` and `merge_navcen` signatures and clock-free behavior
  are unchanged. Callers making operational decisions should migrate to the
  explicit-time APIs; see `NAVCEN_TIME_SEMANTICS.md`. The time-aware path also
  recognizes active `UNUSUFN` notices as immediately unusable; the legacy path's
  pre-existing omission of that code is intentionally preserved.

## [0.31.2] - 2026-07-16

### Fixed

- Included the public merged-SP3 v1 golden fixture inside the published
  `sidereon-core` crate so its integration tests compile from an isolated
  crates.io source archive.

## [0.31.1] - 2026-07-16

### Fixed

- Canonical merged-SP3 identities now normalize accepted negative-zero
  tolerances to positive zero, matching merge execution semantics.
- Merge execution and provenance identity validation now reject the same empty
  system filters and incomplete asserted frame-label sets.
- Added literal cross-interface v1 golden vectors covering the complete merge
  policy, canonical contributor ordering, precedence ordering, and malformed
  inputs.

## [0.31.0] - 2026-07-16

### Added

- Added `Sp3ArtifactIdentity` and `Sp3MergeInputIdentity`, a versioned,
  order-independent stable identity for the complete set of exact SP3
  artifacts and merge controls. The canonical identity binds requested and
  resolved product identities, distributor, product and archive digests and
  lengths, compression, and every merge option while excluding acquisition
  observations, URLs, credentials, and local paths. Mean/median contributor
  enumeration is canonicalized; precedence contributor order is bound as an
  effective policy control.
- Added exact-identity distribution-location derivation so alternate cataloged
  SP3 duration and sampling candidates retain their declared identity instead
  of being reconstructed as the catalog default.

### Compatibility

- This release is additive. Existing SP3 merge and product-location APIs retain
  their prior signatures; exact provenance identity construction is opt-in.

## [0.30.0] - 2026-07-16

### Added

- Added the schema-v3 exact-product cache protocol. Commit records bind the
  complete product identity, explicit distribution source, and SHA-256 digest
  and length of validated product, distributor archive, and provenance bytes.
- Added native Linux/macOS `ExactProductCache` transactions with bounded
  cross-process locking, cryptorandom immutable entries, synchronized files and
  directories, one atomic commit marker, unlocked-reader refresh retry, and
  lock-scoped abandoned-entry cleanup. The ergonomic `sidereon` crate re-exports
  the same API.
- Added `analysis_center` and `format_version` to `ProductIdentity`. Canonical
  identity bytes and portable keys now include every exact identity field.

### Compatibility

- This is a source-breaking identity-model correction: Rust struct literals
  must provide the two new fields. The minor version advances because
  `ProductIdentity` is externally constructible; `cargo-semver-checks` confirms
  a patch release would be incorrect.

### Fixed

- Prevented finite even-count robust-median inputs near `f64` limits from
  overflowing their central-pair addition to infinity.

## [0.29.2] - 2026-07-16

### Added

- Added `validate_exact_product_set`, a sans-IO completion gate for workflows
  that require several exact products. It rejects empty declarations,
  duplicate expected or available identities, missing identities, and
  undeclared identities before dependent processing begins.
- Exact-set comparison uses the complete distributor-independent identity, so
  same-filename products with different prediction tiers remain distinct.
  SP3 observed/predicted timing remains authoritative only through
  `Sp3::prediction_summary()` and its record-level flags.

## [0.29.1] - 2026-07-15

### Fixed

- CODE predicted IONEX direct locations now use AIUB's supported HTTPS
  download endpoint and the exact `CODE/IONO/P1/<year>` or
  `CODE/IONO/P2/<year>` directory selected by the requested prediction tier.
- The directory year is derived from the resolved product identity, including
  a P2 request whose one-day offset crosses into a new year. Exact filenames,
  prediction horizons, and distributor-independent cache identities are
  unchanged; no older date, alternate tier, or provider is substituted.

## [0.29.0] - 2026-07-15

### Added

- Added an exact public GNSS product identity model that keeps product family,
  publisher, solution class, campaign, issue, cadence, coverage date, official
  filename version, format, and prediction horizon separate from distribution.
- Added explicit direct-archive, NASA CDDIS/Earthdata, local-file, and in-memory
  distribution sources, including exact CDDIS SP3 and IONEX locations and
  deterministic source-specific cache paths.
- Added validated product requests and distribution locations so selecting a
  distributor cannot silently change the requested center, tier, issue,
  cadence, date, family, or official filename.

### Compatibility

- This release is additive. Existing product selection, URL generation, and
  cache behavior remain unchanged; the new exact-identity and distribution
  APIs are opt-in.

## [0.28.1] - 2026-07-15

### Fixed

- CODE ultra-rapid SP3 candidates now use AIUB's official HTTPS download
  endpoint while retaining the daily `0000` issue and the dated, alternate,
  and latest-alias filenames. The remaining AIUB-backed catalog entries were
  audited separately and remain unchanged because current and historical trees
  use mixed naming and availability conventions.
- The SP3 validation harness no longer treats access denial or transport
  failure as product absence. Candidate-URL 404/410 statuses are retained in
  reports, while transport failures preserve source, filename, URL, and the
  available HTTP status or network diagnostic.
- Sequential RTK solves with baseline process noise now enforce the exact
  symmetry of the information-form time update. The rank-3 correction is
  symmetric mathematically, but independently evaluated matrix triangles could
  accumulate phase-precision roundoff and destabilize long held-ambiguity arcs.

### Evaluation-bit stability

- Process-noise-enabled sequential RTK updates can move in their last bits when
  the two information-matrix triangles are averaged. The zero-process-noise
  path and public interfaces are unchanged.

## [0.28.0] - 2026-07-13

### Added

- Added deterministic contested-cell outlier rejection, per-cell precedence,
  mixed-cadence SP3 coverage merging, clock-outlier provenance, and an opt-in
  whole-satellite precedence mode.
- Added SP3 per-epoch observed/predicted metadata and the contiguous
  observed-through boundary derived from record flags.
- Added current and alternate ultra-rapid SP3 catalog locations for IGS, CODE,
  ESA, and GFZ.

## [0.27.1] - 2026-07-13

### Fixed

- `lambda_ils_search` now rejects ambiguity values outside the `i64` lattice
  domain before reduction and checks back-transformed candidates before integer
  conversion. Previously, an extreme finite input could saturate to
  `i64::MAX`, overflow canonical rescoring, and return `Ok` with non-finite
  scores and ratio.

### Evaluation-bit stability

- Calls whose ambiguity inputs and back-transformed candidates remain within
  the `i64` lattice domain retain the same LAMBDA arithmetic, candidate
  ordering, scores, and fix decisions. Inputs or internally produced candidates
  outside that domain now return the existing typed `InvalidInput` error.

## [0.27.0] - 2026-07-12

### Added

- `GeoidGrid::from_proj_egm96_gtx` loads the public PROJ EGM96 15-arcminute
  GTX grid. `GeoidGrid::undulation_proj_rad` reproduces PROJ 9.3.0's radian
  indexing and interpolation order with an explicit
  `ProjVgridshiftArithmetic` selection for contracted or separately rounded
  multiply-add evaluation. The new path is pinned against 13,051 public-grid
  reference points; invalid coordinates return typed errors.

### Evaluation-bit stability

- Existing geoid loaders and `GeoidGrid::undulation_rad` retain their previous
  evaluation bits. The new PROJ path has no implicit platform-dependent
  default: callers select fused or separately rounded arithmetic explicitly.

## [0.26.1]

### Security and availability

- RINEX observation parsing now rejects epoch record counts that exceed the
  format's three-character `I3` maximum of 999 before reserving record storage.
  A malformed RINEX 2 epoch could previously pass an effectively unbounded
  count to `Vec::with_capacity`, allowing memory exhaustion or a process abort
  instead of a parse error. `sidereon-core` releases 0.11.1 through 0.26.0,
  inclusive, are affected.

### Evaluation-bit stability

- Valid RINEX inputs retain identical parsed values and evaluation bits. Only
  malformed epochs with an over-width record count change behavior: they now
  return a deterministic parse error before allocation.

## [0.26.0]

### Breaking

- Removed the generic sequential-RTK innovation-screen API and its result
  fields: `InnovationScreenOpts`, `InnovationScreen`,
  `UpdateOpts::innovation_screen`, `EpochUpdate::innovation_screen`,
  `RtkArcEpochSolution::innovation_screen`,
  `ScreenKind::RtkSequentialInnovation`, and
  `ResidualNormRecipe::RtkInverseVarianceInnovation`. The removed mechanism
  divided residuals by measurement variance, omitted predicted-state
  covariance and shared-reference correlation during classification, and
  treated carrier-phase events as ordinary row outliers. Sequential RTK now
  consistently assimilates the complete correlated double-difference block;
  carrier anomalies remain handled by the causal slip/arc lifecycle.
- Removing those variants also compacts the enums' compiler-assigned numeric
  discriminants. Code that casts them to integers will observe
  `ResidualNormRecipe::RtkInverseSigmaResidual` changing from 1 to 0,
  `ResidualNormRecipe::PppInverseSigmaMagnitude` from 2 to 1, and
  `ScreenKind::PppFloatLeaveOneOut` from 3 to 2. These enums do not promise a
  stable numeric representation; 0.26.0 does not preserve unused discriminant
  holes.

### Fixed

- Ionospheric pierce-point evaluation now remains finite when floating-point
  rounding puts a valid near-polar latitude sine just outside `[-1, 1]`.
- The locked dependency graph now uses `crossbeam-epoch` 0.9.20, which fixes
  RUSTSEC-2026-0204.

### Evaluation-bit stability

- The near-polar TEC correction intentionally changes affected pierce-point
  results from non-finite latitude/longitude values to finite values. Existing
  in-range TEC evaluations require no golden re-pin, and the ordinary
  sequential-RTK path remains bit-identical to its former no-screen execution.

## [0.25.0]

### Added

- The `sidereon` facade root now exposes the CRINEX encoder convenience
  `encode_crinex`, matching the existing lower core module and bindings.
- The `sidereon` facade root now re-exports existing Sun/Moon azimuth/elevation
  helpers, geodetic/topocentric transform helpers, TLE look-angle and
  ground-track helpers, and Doppler shift helpers that were already available
  through lower core modules.

## [0.24.0]

### Changed

- ARAIM now returns an unavailable `AraimResult` with `available: false` when
  geometry cannot support the integrity budget, instead of returning
  `UnmonitorableFaultMass`.

## [0.23.0]

### Added

- RTCM 3 broadcast ephemeris decode/encode and solver conversion for Galileo
  1045/1046, BeiDou 1042, and QZSS 1044. Galileo 1046 is covered by a real HAS
  IDD capture propagated against a CNES/CLS ultra-rapid SP3 trim; BeiDou 1042,
  QZSS 1044, and Galileo F/NAV 1045 are covered by a real BKG BCEP capture
  propagated against matching CNES/CLS and QZSS ultra-rapid SP3 trims.
- Static PPP float and fixed solve configs now accept an optional
  `elevation_cutoff_deg`; when set, observations below the seed-position
  elevation cutoff are removed before ambiguity ids, residual rows, normal rows,
  and fixed ambiguity search are assembled. `None` preserves the existing
  observation set.
- optional tropospheric horizontal gradient estimation for static PPP (off by
  default).
- static multi-epoch positioning (`solve_static`) is now public, with
  covariance, leave-one-out redundancy diagnostics, and robust weighting.

- Static PPP float and fixed solutions add temporal-correlation covariance
  reporting: `temporal_position_covariance`,
  `temporal_position_covariance_scale_factor`, and `temporal_correlation`.

## [0.22.0]

### Added

- Static PPP float and fixed solutions expose posterior receiver-position
  covariance in ECEF and ENU coordinates through `PositionCovariance`, plus the
  raw posterior unit-variance factor and the applied covariance scale factor
  (which equals that factor). Every solver in the library now reports position
  covariance.
  The estimator pools lag-1 post-fit residual autocorrelation by satellite arc
  and observable, reports AR(1) effective sample count and decorrelation time,
  and keeps the existing posterior-scaled covariance fields unchanged.
- SP3 multi-center merge coordinate-label reconciliation: caller-asserted label
  equivalence and catalog Helmert reconciliation between known ITRF/IGS
  realizations, with merge-report audit fields for the selected method, affected
  records, published parameters, rates, provenance, and catalog direction.
  Strict label matching remains the default; unresolvable mismatches still fail.
- RTCM MSM stream-to-SPP conversion for live workflows: `RtcmSppEpochInputs` and
  `spp_inputs_from_rtcm_msm` assemble RTCM MSM observations into the same
  per-epoch solve input shape used by RINEX replay.
- Allocation-free warm hot path for high-rate serving:
  `emission_media_batch_at_j2000_s_into` writes the correction bundle into
  caller buffers, and `EmissionMediaReceiverContext` plus
  `emission_media_batch_at_j2000_s_with_receiver_context_into` cache the
  per-receiver setup so a staged, repeated single-call path allocates nothing.
  Results are bit-identical to the allocating form (gated across the fixture
  sweep). Staged precise-ephemeris interpolants gain `EphemerisSource` impls,
  with a bit-identity gate pinning the staged-interpolant SPP solve to the raw
  SP3 solve.

### Changed

- Static PPP eliminates per-epoch receiver clocks from the normal equations and
  back-substitutes them after solving the reduced static system, making
  day-length arcs tractable without changing the public clock output.
- Static PPP result covariance is multiplied by the posterior residual variance
  factor, with the unscaled formal covariance retained for callers.
- PPP GF/MW cycle-slip splitting confirms GF/MW-only events before creating new
  ambiguity states, while LLI and data-gap splits remain immediate.

### Breaking

- `FloatSolution` and `FixedSolution` in precise positioning gained required
  `position_covariance`, formal covariance, and posterior variance scale
  fields. Callers constructing these structs directly must populate them;
  callers only reading results are unaffected.

## [0.21.0]

### Added

- Loose GNSS/INS field-mode options: stationary ZUPT/ZARU pseudo-updates with
  a configurable accel and gyro magnitude window, wheeled-vehicle
  non-holonomic lateral and vertical velocity constraints, and
  per-fix-status GNSS covariance weighting for single, float, and fixed
  updates. The inertial filter config also accepts a fixed IMU-to-body
  direction-cosine matrix for callers that do not pre-rotate IMU samples.
- Standalone first-fix velocity matching helpers for GNSS outage spans,
  including `velocity_match_outage_to_state` for blending an outage segment to
  a caller-supplied post-update endpoint instead of only to the raw GNSS fix.
- RINEX-to-SPP assembly helpers that convert parsed observation epochs plus a
  broadcast or precise ephemeris context into per-epoch `SolveInputs`, with a
  serial batch solve convenience that preserves per-epoch solve errors.
- Static reference-station RINEX solve that composes code-DGNSS and carrier RTK
  modes, returning one station coordinate with covariance, fix status, and
  per-epoch diagnostics.

### Changed

- Fusion state checkpoints use codec version 4 and still read earlier v1-v3
  streams. Checkpoints now preserve the stationary-detector window plus the
  last stationary and non-holonomic pseudo-update epochs, so restored filters
  keep detector state and duplicate-update guards.
- `GnssFixMeasurement` now carries public `fix_status`; JSON
  `SerializableLooseMeasurement` defaults the field for older payloads.
- RTK `FloatBaselineSolution` and `FixedBaselineSolution` now expose
  `baseline_covariance_m2`; computing that covariance can surface
  `SingularGeometry` on degenerate final normal equations.
- Fusion RTS histories accept same-epoch predicted/updated checkpoints for
  measurement-only updates, synthesize an identity transition for those
  updates, and permit zero-duration smoothing transitions.
- Static reference-station selection now prefers fixed carrier RTK, then code
  DGNSS, then float carrier fallback; reports keep the fixed-solution
  measurement count, label fixed-mode failures correctly, carry typed
  per-mode errors, and format all-mode failures by mode instead of dumping
  debug structs.
- Code-DGNSS covariance now accounts for both rover and reference code noise,
  including the multi-epoch static reference-station path.
- Stationary ZUPT/ZARU and non-holonomic pseudo-updates no longer inherit GNSS
  IGG-III measurement reweighting or Yang prediction-adaptation settings.
- Tight-coupling range-rate gyro-bias rows now honor `imu_to_body_dcm` for
  non-identity IMU mounting.
- Real BKG/IGS-IP SSRA03IGS0 SSR integration fixture covering GPS, GLONASS,
  Galileo, and BeiDou RTCM SSR orbit, clock, and code-bias decode. The test
  validates IODE-matched GPS SSR-corrected broadcast satellite positions
  against the IGS ultra-rapid SP3 with a non-vacuous broadcast-only error
  margin.

## [0.20.0]

### Added

- RINEX RTK arc builders as library API: rover and base observations plus
  ephemeris and base coordinates in, double-differenced carrier-phase arcs
  built by the library, static float and wide-lane fixed baselines out with
  fix status. On the real WTZR/WTZZ station pair the fixed
  baseline lands within 2.8 mm of the published ITRF antenna-reference-point
  baseline (float: 8.3 mm).

- SSR and Galileo HAS corrections now drive the PPP solve: an SSR-corrected
  ephemeris provider applies orbit and clock corrections over broadcast
  ephemeris with strict IODE matching, update-interval staleness handling, and
  explicit antenna-phase-center versus center-of-mass reference handling; RTCM
  SSR code and phase biases for GPS and Galileo decode into the correction
  store and apply in the PPP measurement model. On the end-to-end fixture the
  SSR-corrected solve closes an 11.6 m broadcast-only error to below 0.1 mm
  against the SP3-backed reference.

## [0.19.0]

### Changed

- The fusion smoother's transition-combination step uses a specialized square
  matrix product, making fixed-interval smoothing tractable over histories
  recorded at inertial sample rates (found during the deep-urban field
  rematch).
- Correction to an earlier draft of this entry: a one-ULP evaluation shift on
  Earth-orientation-chain paths appeared mid-cycle from the tide-force wiring
  and was reversed by the station-displacement refactor before release. Net
  evaluation bits for these surfaces are UNCHANGED relative to 0.18.0.

### Added

- Station displacement corrections now have a public `tides` entry that accepts
  ITRF/ECEF or WGS84 geodetic station positions, UTC epochs, per-epoch IERS
  polar motion, and optional caller-supplied BLQ ocean-loading coefficients.
  The scalar and batch APIs return component-resolved ITRF/ECEF displacements
  for solid Earth tide, pole tide, and ocean tidal loading. BLQ parsing supports
  standard Bos-Scherneck/HARDISP six-row station blocks and reports typed errors
  for unsupported constituents.
- Solid Earth tide and solid Earth pole tide propagation forces. The solid
  Earth tide force ships the IERS 2010 Chapter 6 Step 1 frequency-independent
  anelastic Love-number `Cnm`/`Snm` corrections from Sun and Moon positions,
  including degree 3 terms and degree 4 `k+` terms from degree-2 tides. Step 2
  frequency-dependent constituent corrections are documented as a follow-up.
  The pole tide force uses polar-motion samples from a series-backed
  body-fixed provider and the IERS 2010 mean pole model, and both forces are
  opt-in builder components.

## [0.18.0]

### Added

- Loose GNSS updates can opt into IGG-III measurement variance inflation and a
  Yang two-segment prediction adaptive factor. The prediction factor is gated
  by a Jiang-Zhang Mahalanobis measurement-outlier check so measurement faults
  use measurement reweighting rather than innovation-driven covariance scaling.
- Fusion RTS fixed-interval smoothing over recorded error-state histories,
  with recorded forward-pass transitions, predicted and updated checkpoints,
  smoothed covariances, and loose/tight measurement-agnostic entry points.
- Simulator-backed field-behavior pins for loose fusion smoothing, outage
  coast, and low-satellite tight consistency.

## [0.17.0]

### Fixed

- Tight GNSS C1C and carrier-phase code-row prediction now uses the same
  measured-pseudorange transmit-time model as SPP, removing centimetre-level
  frozen-state residual differences from the prior observable transmit-time
  approximation.
- Sample-backed SP3 interpolation now reconstructs the whole-second node axis
  from the split epoch before reducing it to continuous J2000 seconds, with
  only an epoch-ULP bound for accepting whole-second candidates. An earlier
  attempt used an absolute snap that did not fire for real converted epochs,
  which land one `f64` ULP below affected record seconds; the record-epoch
  oracle now runs a conversion-path fixture on both construction paths.

### Changed

- IONEX slant-delay evaluation now reports out-of-coverage epochs and
  pierce-point latitude or longitude as typed errors by default instead of a
  silent hold. Callers can opt into the legacy hold behavior with
  `IonexCoveragePolicy::Hold`, which returns an explicit status marker, and the
  new batch result helper reports coverage per element.

## [0.16.1]

### Fixed

- SP3 interpolation on the parsed-product path now uses an exact parsed
  J2000-second epoch axis for record nodes, preventing one-second node bucketing
  errors at affected 45-minute cadence boundaries. Record-epoch positions and
  clocks are gated directly against public SP3 text records, including the
  cached batch path. (A clock quantization initially reported alongside this
  was traced to the reporting consumer's own time conversion, not to this
  library; the record-epoch clock oracle it prompted remains, at 5e-13 s
  against the file text.)

## [0.11.1]

### Added

- GNSS observation QC now computes teqc-style multipath (MP1/MP2 RMS) per
  satellite and per constellation with per-arc moving-average bias removal,
  matching teqc `+qc` to sub-micrometer on a real captured stream; a
  receiver clock-jump detector; and an aggregate per-constellation cycle-slip
  tally over the existing dual-frequency slip detector.
- QC report renderers: a fixed-width teqc-style text summary, an HTML summary,
  and JSON serialization of the full `ObservationQcReport`.
- RINEX 2.x observation-file ingest into the shared canonical observation IR,
  so RINEX 2 and CRINEX 1.0 archives parse and flow through QC and lint
  unchanged.
- Fuzz targets for the space-weather CSV/txt parser, the RINEX QC repair
  round-trip, and the EGM96 DTED grid parser.

### Changed

- Rust, Python, C, WASM, and Elixir interfaces expose the new QC surface
  (multipath, clock jumps, cycle-slip tally, and the report renderers)
  with uniform parity.

## [0.11.0]

### Added

- Core 6x6 orbit covariance transport with frame-labeled nodes,
  RTN acceleration process noise, caller-supplied transport segments,
  PSD-safe Log-Cholesky interpolation, covariance unit conversion helpers, and
  TCA Pc integration for propagated covariances.
- Space-weather ingestion: CSSI space-weather CSV and txt parsing with a
  time-indexed table, NRLMSISE-00 selection conventions (previous-day F10.7,
  81-day centered average, daily and 3-hourly Ap), format-faithful
  serializers, a CelesTrak data-catalog entry, and a `SpaceWeatherSource`
  hook feeding atmospheric drag and orbital-decay estimation.
- GNSS observation quality control: per-satellite and per-signal completeness,
  gap, and signal-strength summaries over RINEX observation data, plus a RINEX
  lint and repair pass with typed finding codes, cross-checked against an
  independent extraction oracle on real IGS stations. (Multipath, cycle-slip,
  and clock-jump metrics land in 0.11.1.)
- RTCM MSM carrier-phase lock-time indicator to RINEX loss-of-lock indicator
  derivation: DF402 and DF407 lock-time bucket tables, conservative
  decrease detection with same-bucket ambiguity handling, half-cycle ambiguity
  mapping, and a per-signal lock-time tracker, cross-checked against an
  independent RTKLIB convbin decode of a real MSM stream. Adds RTCM stream
  decode diagnostics and typed truncation classification.
- NTRIP sans-IO protocol: a caster handshake and streaming state machine
  (request builder, response classification, chunked and sourcetable decoding,
  GGA position feed policy) with no transport in the core, plus idiomatic
  streaming clients in the Python and Elixir interfaces.
- NMEA 0183 support: a forgiving sentence parser, an epoch accumulator, and a
  GGA writer over a format-agnostic representation.
- CNAV and RINEX-4 broadcast evaluation: CNAV and CNAV2 clock and orbit
  parameters, user range accuracy and inter-signal correction accessors,
  mixed broadcast-store selection, and a lenient navigation parse that reports
  skipped blocks.
- TLE mean-element fitting: fit SGP4 elements to a span of states on the
  shared trust-region least-squares engine, with observability diagnostics,
  epoch selection, and observation weighting. NDM epochs gain femtosecond
  precision through a single shared parser.
- Geoid undulation evaluation matching PROJ on the EGM96 15-arcminute grid:
  node-registered bilinear interpolation with antimeridian and pole handling,
  batch lookup, and orthometric to ellipsoidal height conversion, pinned to
  PROJ-computed reference values.

### Changed

- `Covariance6Error` now includes interpolation-specific
  `NotFactorizable` and `InvalidInterpolationParameter` variants, and
  propagated-covariance TCA option structs now carry `process_noise`.
  Exhaustive matches and struct literals may need source updates.
- Rust, Python, C, WASM, and Elixir interfaces expose uniform capability parity
  for the 0.11.0 surface.

## [0.10.1]

### Fixed

- DTED ten-degree block directories now follow the layout production stores
  use: the hemisphere letter comes from the tile index and the magnitude is the
  truncated absolute value, so `n36_w107` buckets under `n30_w100/`,
  `n32_w118` under `n30_w110/`, and `s01_w001` under `s00_w000/` (with `n00`
  and `s00` kept distinct). The previous flooring convention mis-bucketed
  every western and southern index that was not an exact multiple of ten,
  making caches invisible to existing tile stores. Tile naming itself is
  unchanged. A cache directory populated by 0.10.0 can be migrated by moving
  the affected tiles into the corrected block directories, or simply
  regenerated. The derivation is validated against an observed 888-tile
  listing captured from a production-style store.
- `PreciseEphemerisSamples::from_samples` now rejects a sample epoch whose
  derived J2000 seconds is not finite and a finite clock offset that overflows
  to a non-finite value in native microseconds, instead of poisoning the
  interpolation node axis or emitting non-finite clock values downstream.

## [0.10.0]

### Added

- Astrodynamics coverage for anomaly conversions, analytic Kepler propagation,
  equinoctial and modified-equinoctial elements, solar beta angle,
  RIC/RTN/LVLH relative frames, Clohessy-Wiltshire motion, angular separation,
  position angle, general body observation, almanac events, atmospheric drag
  force, orbital decay, source-agnostic ephemeris grid sampling, and
  terrain/DTED lookup.
- GNSS DCB/OSB bias ingestion, SBAS augmentation with decode and corrected SPP,
  SSR/HAS real-time corrections, and robust SPP with a fault
  detection/exclusion driver.
- Cache-first data acquisition support for SP3, IONEX, CLK, NAV, and SRTM
  terrain to DTED products, using a single sans-IO core catalog and bit-exact
  hgt to DTED conversion.

### Changed

- Rust, Python, C, WASM, and Elixir interfaces now expose uniform capability
  parity for the 0.10.0 surface.
- GNSS constellation labels now use conventional styling: GPS, GLONASS, Galileo,
  BeiDou, QZSS, NavIC, and SBAS.

## [0.1.0]

Initial release.

- SGP4/SDP4 propagation (Vallado port), TLE and OMM (KVN/XML/JSON) parsing.
- Coordinate and time transforms (TEME/GCRS/ITRS/geodetic/topocentric, leap
  seconds, UT1), Sun/Moon ephemeris, solid-earth tides.
- RINEX navigation/observation/clock and CRINEX parsing, SP3 load and merge,
  ANTEX antenna corrections, broadcast and precise ephemeris evaluation.
- GNSS positioning: SPP (with robust estimation), RTK (LAMBDA ambiguity
  resolution, dual-frequency, multi-GNSS), and static PPP.
- Carrier-phase combinations and cycle-slip detection, DOP, visibility and
  pass prediction, velocity/Doppler, and observation quality weighting.
- Conjunction assessment and collision probability.
