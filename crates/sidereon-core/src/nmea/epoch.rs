use std::collections::BTreeSet;

use super::{
    Gga, Gll, Gsa, Gst, Gsv, GsvSatellite, NmeaBody, NmeaDate, NmeaSatNumber, NmeaSentence,
    NmeaSignalId, NmeaTalker, NmeaTime, Rmc, Vtg, Zda,
};
use crate::format::{Diagnostics, Parsed, RecordRef, Warning, WarningKind};

const RETAINED_CAP: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
/// One accumulated NMEA epoch, closed by an anchor-time, GSV-cycle, or sentence-budget boundary.
/// Singleton sentence kinds retain their first body, while GSA and GSV data are accumulated in their respective entry and group vectors.
pub struct EpochSnapshot {
    /// Time from the first GGA, RMC, GLL, GST, or ZDA sentence in the epoch that carries one.
    pub time_of_day: Option<NmeaTime>,
    /// The explicit RMC/ZDA date, or the date carried from [`NmeaAccumulator::with_date`] or an earlier dated sentence.
    /// A differing explicit date replaces the prior value and records a mismatch warning; a backward transition over 12 hours advances the carried date for the next epoch.
    pub date: Option<NmeaDate>,
    /// The first GGA body attached to the epoch; later GGA bodies are omitted and record a mismatch warning.
    pub gga: Option<Gga>,
    /// The first RMC body attached to the epoch; later RMC bodies are omitted and record a mismatch warning.
    pub rmc: Option<Rmc>,
    /// The first GLL body attached to the epoch; later GLL bodies are omitted and record a mismatch warning.
    pub gll: Option<Gll>,
    /// The first GST body attached to the epoch; later GST bodies are omitted and record a mismatch warning.
    pub gst: Option<Gst>,
    /// The first VTG body attached to the epoch; later VTG bodies are omitted and record a mismatch warning.
    pub vtg: Option<Vtg>,
    /// The first ZDA body attached to the epoch; later ZDA bodies are omitted and record a mismatch warning.
    pub zda: Option<Zda>,
    /// GSA bodies accumulated as [`GsaEntry`] values, with known systems deduplicated and unknown-system entries retained.
    pub gsa: Vec<GsaEntry>,
    /// GSV pages accumulated as [`GsvGroup`] values, grouped by talker and optional signal.
    pub gsv: Vec<GsvGroup>,
    /// Number of accepted sentences attached to the epoch, including duplicate bodies and sentences without a time.
    pub sentence_count: usize,
    /// Assembly warnings for duplicate or inconsistent epoch data, GSV sequence/count errors, near-midnight backward time, and the sentence budget.
    /// Parser skips and warnings are kept in [`NmeaChunkOutput::diagnostics`] instead.
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone, PartialEq)]
/// One decoded GSA body together with the optional GNSS system context used to distinguish it within an [`EpochSnapshot`].
pub struct GsaEntry {
    /// The system copied from [`Gsa::system`], or None when parsing could not resolve a system context.
    pub system: Option<crate::GnssSystem>,
    /// The cloned GSA body, including its sentence-order satellite list and optional DOP values.
    pub gsa: Gsa,
}

#[derive(Debug, Clone, PartialEq)]
/// The accumulated GSV pages for one talker and optional signal within an [`EpochSnapshot`].
pub struct GsvGroup {
    /// The talker copied from the NMEA sentence carrying the GSV page.
    pub talker: NmeaTalker,
    /// The optional signal copied from [`Gsv::signal`], used with `talker` to select the page group.
    pub signal: Option<NmeaSignalId>,
    /// The first page’s optional claimed satellite count, replaced by the current page’s count after a sequence mismatch.
    /// When the group completes, this value is compared with the number of listed satellite numbers.
    pub claimed_in_view: Option<u16>,
    /// Satellite groups from the first page, extended in page order for an expected sequence or replaced after a sequence mismatch.
    pub satellites: Vec<GsvSatellite>,
    /// Whether the latest attached page number equals its total page count.
    /// A new page 1 for the same complete talker/signal group closes the current epoch before attachment.
    pub complete: bool,
}

impl EpochSnapshot {
    fn empty(date: Option<NmeaDate>) -> Self {
        Self {
            time_of_day: None,
            date,
            gga: None,
            rmc: None,
            gll: None,
            gst: None,
            vtg: None,
            zda: None,
            gsa: Vec::new(),
            gsv: Vec::new(),
            sentence_count: 0,
            diagnostics: Diagnostics::new(),
        }
    }

    /// Returns a WGS-84 position using GGA first, then RMC, then GLL.
    /// GGA requires both coordinates, MSL altitude, and geoid separation and converts their height sum to ellipsoidal height; RMC and GLL use zero height, and any missing required field or invalid WGS-84 construction returns `None`.
    pub fn position(&self) -> Option<crate::Wgs84Geodetic> {
        if let Some(gga) = &self.gga {
            let latitude = gga.latitude?;
            let longitude = gga.longitude?;
            let altitude_msl_m = gga.altitude_msl_m?;
            let geoid_separation_m = gga.geoid_separation_m?;
            return crate::Wgs84Geodetic::new(
                latitude.radians(),
                longitude.radians(),
                altitude_msl_m + geoid_separation_m,
            )
            .ok();
        }
        if let Some(rmc) = &self.rmc {
            return crate::Wgs84Geodetic::new(
                rmc.latitude?.radians(),
                rmc.longitude?.radians(),
                0.0,
            )
            .ok();
        }
        let gll = self.gll.as_ref()?;
        crate::Wgs84Geodetic::new(gll.latitude?.radians(), gll.longitude?.radians(), 0.0).ok()
    }

    /// Converts the stored date and time to a UTC [`crate::astro::time::Instant`].
    /// Nanoseconds are added to the whole seconds, and `None` is returned when either field is absent or the civil-time conversion rejects the values.
    pub fn instant_utc(&self) -> Option<crate::astro::time::Instant> {
        let date = self.date?;
        let time = self.time_of_day?;
        let second = f64::from(time.second) + f64::from(time.nanos) * 1.0e-9;
        crate::astro::time::Instant::from_utc_civil(
            i32::from(date.year),
            i32::from(date.month),
            i32::from(date.day),
            i32::from(time.hour),
            i32::from(time.minute),
            second,
        )
        .ok()
    }

    /// Returns the first position dilution of precision supplied by a GSA entry, in entry order.
    /// Returns `None` when no GSA entry has a PDOP value.
    pub fn pdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.pdop)
    }

    /// Returns the first horizontal dilution of precision supplied by a GSA entry, in entry order.
    /// Returns `None` when no GSA entry has an HDOP value.
    pub fn hdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.hdop)
    }

    /// Returns the first vertical dilution of precision supplied by a GSA entry, in entry order.
    /// Returns `None` when no GSA entry has a VDOP value.
    pub fn vdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.vdop)
    }

    /// Iterates over GSA satellite numbers in entry order and within each entry’s sentence order.
    /// The iterator borrows the stored values and does not remove duplicates across entries.
    pub fn used_satellites(&self) -> impl Iterator<Item = &NmeaSatNumber> {
        self.gsa
            .iter()
            .flat_map(|entry| entry.gsa.satellites.iter())
    }

    /// Counts unique `(resolved identity, raw number)` pairs listed by the GSV satellite groups.
    /// Entries without a satellite number and the pages’ claimed counts are ignored.
    pub fn satellites_in_view(&self) -> usize {
        let mut seen = BTreeSet::new();
        for group in &self.gsv {
            for sat in &group.satellites {
                if let Some(number) = sat.sat_number {
                    seen.insert((number.resolved, number.raw));
                }
            }
        }
        seen.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OpenEpoch {
    snapshot: EpochSnapshot,
    gsv_progress: Vec<GsvProgress>,
}

#[derive(Debug, Clone, PartialEq)]
struct GsvProgress {
    talker: NmeaTalker,
    signal: Option<NmeaSignalId>,
    total: u8,
    next_expected: u8,
}

#[derive(Debug, Clone, PartialEq)]
/// Streaming accumulator for one open NMEA epoch and the bytes waiting for a line terminator.
/// It also carries dates across undated sentences, enforces a sentence budget, and numbers input lines for parser diagnostics.
pub struct NmeaAccumulator {
    current: Option<OpenEpoch>,
    carried_date: Option<NmeaDate>,
    previous_anchor: Option<NmeaTime>,
    max_sentences_per_epoch: usize,
    retained: Vec<u8>,
    next_line: usize,
}

impl Default for NmeaAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl NmeaAccumulator {
    /// Creates an accumulator with no open epoch or carried date, a 256-sentence epoch limit, an empty byte remainder, and next line number 1.
    pub fn new() -> Self {
        Self {
            current: None,
            carried_date: None,
            previous_anchor: None,
            max_sentences_per_epoch: 256,
            retained: Vec::new(),
            next_line: 1,
        }
    }

    /// Creates the default accumulator with `date` available to epochs whose sentences do not carry a date.
    pub fn with_date(date: NmeaDate) -> Self {
        Self {
            carried_date: Some(date),
            ..Self::new()
        }
    }

    /// Sets the per-epoch sentence budget, clamping values below 16 to 16.
    pub fn with_max_sentences_per_epoch(mut self, max: usize) -> Self {
        self.max_sentences_per_epoch = max.max(16);
        self
    }

    /// Attaches one already parsed sentence and returns a snapshot when the prior epoch closes.
    /// A close occurs when both time anchors differ, a completed GSV cycle restarts, or the sentence budget is reached; the incoming sentence is then attached to a newly opened epoch, and a budget close records a mismatch warning.
    pub fn push(&mut self, sentence: &NmeaSentence) -> Option<EpochSnapshot> {
        let incoming_time = sentence_time(sentence);
        let mut new_epoch_warning = false;
        let mut completed = None;

        if let Some(current) = &mut self.current {
            if let (Some(current_time), Some(incoming)) =
                (current.snapshot.time_of_day, incoming_time)
            {
                if current_time.key() != incoming.key() {
                    completed = self.current.take().map(|epoch| epoch.snapshot);
                    self.apply_boundary_date_policy(current_time, incoming, &mut new_epoch_warning);
                }
            }
        }

        if completed.is_none() && self.gsv_cycle_boundary(sentence) {
            completed = self.current.take().map(|epoch| epoch.snapshot);
        }

        if completed.is_none()
            && self
                .current
                .as_ref()
                .is_some_and(|epoch| epoch.snapshot.sentence_count >= self.max_sentences_per_epoch)
        {
            if let Some(mut epoch) = self.current.take() {
                epoch.snapshot.diagnostics.push_warning(Warning {
                    at: RecordRef::default(),
                    kind: WarningKind::Mismatch,
                });
                completed = Some(epoch.snapshot);
            }
        }

        if self.current.is_none() {
            self.current = Some(OpenEpoch {
                snapshot: EpochSnapshot::empty(self.carried_date),
                gsv_progress: Vec::new(),
            });
            if new_epoch_warning {
                self.warn_current();
            }
        }
        self.attach(sentence);
        completed
    }

    /// Buffers input until LF, CR, or CRLF, parses each complete line, and returns that chunk’s parsed sentences, completed snapshots, and line-numbered parser diagnostics.
    /// If an unterminated buffer exceeds 1024 bytes, it is discarded and reported as a “sentence over length cap” skip.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> NmeaChunkOutput {
        self.retained.extend_from_slice(chunk);
        let mut output = NmeaChunkOutput::default();
        loop {
            let Some((pos, term_len)) = next_line_end(&self.retained) else {
                if self.retained.len() > RETAINED_CAP {
                    super::push_error_skip_at(
                        &mut output.diagnostics,
                        super::NmeaError::NotFramed {
                            reason: "sentence over length cap",
                        },
                        RecordRef::at_line(self.next_line),
                    );
                    self.retained.clear();
                    self.next_line += 1;
                }
                break;
            };
            let line = self.retained.drain(..pos).collect::<Vec<_>>();
            self.retained.drain(..term_len);
            let line_no = self.next_line;
            self.next_line += 1;
            push_line(self, &line, line_no, &mut output);
        }
        output
    }

    /// Processes a nonempty unterminated remainder as one final line, then returns the final open snapshot, if any.
    /// The temporary output produced while parsing that remainder is discarded, including any snapshot completed during that processing.
    pub fn finish(&mut self) -> Option<EpochSnapshot> {
        if !self.retained.is_empty() {
            let line = std::mem::take(&mut self.retained);
            let mut output = NmeaChunkOutput::default();
            push_line(self, &line, self.next_line, &mut output);
            self.next_line += 1;
        }
        self.current.take().map(|epoch| epoch.snapshot)
    }

    /// Returns the number of bytes currently buffered without a line terminator.
    pub fn retained_len(&self) -> usize {
        self.retained.len()
    }

    fn attach(&mut self, sentence: &NmeaSentence) {
        let time = sentence_time(sentence);
        let date = sentence_date(sentence);
        if let Some(date) = date {
            self.carried_date = Some(date);
        }
        let Some(epoch) = self.current.as_mut() else {
            return;
        };
        if epoch.snapshot.time_of_day.is_none() {
            epoch.snapshot.time_of_day = time;
        }
        if let Some(date) = date {
            if epoch.snapshot.date.is_some_and(|existing| existing != date) {
                epoch.snapshot.diagnostics.push_warning(Warning {
                    at: RecordRef::default(),
                    kind: WarningKind::Mismatch,
                });
            }
            epoch.snapshot.date = Some(date);
        } else if epoch.snapshot.date.is_none() {
            epoch.snapshot.date = self.carried_date;
        }
        epoch.snapshot.sentence_count += 1;

        match &sentence.body {
            NmeaBody::Gga(gga) => attach_single(
                &mut epoch.snapshot.gga,
                gga,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Rmc(rmc) => attach_single(
                &mut epoch.snapshot.rmc,
                rmc,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Gll(gll) => attach_single(
                &mut epoch.snapshot.gll,
                gll,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Gst(gst) => attach_single(
                &mut epoch.snapshot.gst,
                gst,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Vtg(vtg) => attach_single(
                &mut epoch.snapshot.vtg,
                vtg,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Zda(zda) => attach_single(
                &mut epoch.snapshot.zda,
                zda,
                &mut epoch.snapshot.diagnostics,
            ),
            NmeaBody::Gsa(gsa) => attach_gsa(&mut epoch.snapshot, gsa),
            NmeaBody::Gsv(gsv) => attach_gsv(
                sentence.talker,
                &mut epoch.snapshot,
                &mut epoch.gsv_progress,
                gsv,
            ),
        }
    }

    fn warn_current(&mut self) {
        if let Some(current) = &mut self.current {
            current.snapshot.diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
    }

    fn gsv_cycle_boundary(&self, sentence: &NmeaSentence) -> bool {
        let NmeaBody::Gsv(gsv) = &sentence.body else {
            return false;
        };
        if gsv.message_number != 1 {
            return false;
        }
        self.current.as_ref().is_some_and(|epoch| {
            epoch.snapshot.gsv.iter().any(|group| {
                group.talker == sentence.talker && group.signal == gsv.signal && group.complete
            })
        })
    }

    fn apply_boundary_date_policy(
        &mut self,
        current: NmeaTime,
        incoming: NmeaTime,
        warning: &mut bool,
    ) {
        self.previous_anchor = Some(current);
        let current_ns = time_of_day_ns(current);
        let incoming_ns = time_of_day_ns(incoming);
        if current_ns > incoming_ns {
            let delta = current_ns - incoming_ns;
            if delta > 43_200_000_000_000 {
                if let Some(date) = self.carried_date {
                    self.carried_date = Some(date.next_day());
                }
            } else {
                *warning = true;
            }
        }
    }
}

fn attach_single<T: Clone>(slot: &mut Option<T>, value: &T, diagnostics: &mut Diagnostics) {
    if slot.is_none() {
        *slot = Some(value.clone());
    } else {
        diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::Mismatch,
        });
    }
}

fn attach_gsa(snapshot: &mut EpochSnapshot, gsa: &Gsa) {
    if let Some(existing) = snapshot
        .gsa
        .iter()
        .find(|entry| entry.system.is_some() && entry.system == gsa.system)
    {
        if existing.gsa.pdop != gsa.pdop
            || existing.gsa.hdop != gsa.hdop
            || existing.gsa.vdop != gsa.vdop
        {
            snapshot.diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
        snapshot.diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::Mismatch,
        });
        return;
    }
    if let Some(first) = snapshot.gsa.first() {
        if differing(first.gsa.pdop, gsa.pdop)
            || differing(first.gsa.hdop, gsa.hdop)
            || differing(first.gsa.vdop, gsa.vdop)
        {
            snapshot.diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
    }
    snapshot.gsa.push(GsaEntry {
        system: gsa.system,
        gsa: gsa.clone(),
    });
}

fn attach_gsv(
    talker: NmeaTalker,
    snapshot: &mut EpochSnapshot,
    progress: &mut Vec<GsvProgress>,
    gsv: &Gsv,
) {
    let group_index = snapshot
        .gsv
        .iter()
        .position(|group| group.talker == talker && group.signal == gsv.signal);
    let progress_index = progress
        .iter()
        .position(|entry| entry.talker == talker && entry.signal == gsv.signal);
    match (group_index, progress_index) {
        (Some(group_index), Some(progress_index)) => {
            let expected = progress[progress_index].next_expected;
            let total = progress[progress_index].total;
            if gsv.total_messages != total || gsv.message_number != expected {
                snapshot.diagnostics.push_warning(Warning {
                    at: RecordRef::default(),
                    kind: WarningKind::Mismatch,
                });
                snapshot.gsv[group_index].satellites = gsv.satellites.clone();
                snapshot.gsv[group_index].claimed_in_view = gsv.satellites_in_view;
                snapshot.gsv[group_index].complete = gsv.message_number == gsv.total_messages;
                progress[progress_index] = GsvProgress {
                    talker,
                    signal: gsv.signal,
                    total: gsv.total_messages,
                    next_expected: gsv.message_number.saturating_add(1),
                };
            } else {
                snapshot.gsv[group_index]
                    .satellites
                    .extend(gsv.satellites.clone());
                snapshot.gsv[group_index].complete = gsv.message_number == total;
                progress[progress_index].next_expected = expected.saturating_add(1);
            }
            if snapshot.gsv[group_index].complete {
                check_gsv_count(snapshot, group_index);
            }
        }
        _ => {
            snapshot.gsv.push(GsvGroup {
                talker,
                signal: gsv.signal,
                claimed_in_view: gsv.satellites_in_view,
                satellites: gsv.satellites.clone(),
                complete: gsv.message_number == gsv.total_messages,
            });
            progress.push(GsvProgress {
                talker,
                signal: gsv.signal,
                total: gsv.total_messages,
                next_expected: gsv.message_number.saturating_add(1),
            });
            if snapshot.gsv.last().is_some_and(|group| group.complete) {
                let index = snapshot.gsv.len() - 1;
                check_gsv_count(snapshot, index);
            }
        }
    }
}

fn check_gsv_count(snapshot: &mut EpochSnapshot, group_index: usize) {
    let group = &snapshot.gsv[group_index];
    if let Some(claimed) = group.claimed_in_view {
        let listed = group
            .satellites
            .iter()
            .filter(|sat| sat.sat_number.is_some())
            .count();
        if usize::from(claimed) != listed {
            snapshot.diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
    }
}

fn differing(a: Option<f64>, b: Option<f64>) -> bool {
    matches!((a, b), (Some(a), Some(b)) if a != b)
}

fn sentence_time(sentence: &NmeaSentence) -> Option<NmeaTime> {
    match &sentence.body {
        NmeaBody::Gga(gga) => gga.time,
        NmeaBody::Rmc(rmc) => rmc.time,
        NmeaBody::Gll(gll) => gll.time,
        NmeaBody::Gst(gst) => gst.time,
        NmeaBody::Zda(zda) => zda.time,
        NmeaBody::Gsa(_) | NmeaBody::Gsv(_) | NmeaBody::Vtg(_) => None,
    }
}

fn sentence_date(sentence: &NmeaSentence) -> Option<NmeaDate> {
    match &sentence.body {
        NmeaBody::Rmc(rmc) => rmc.date,
        NmeaBody::Zda(zda) => zda.date,
        _ => None,
    }
}

fn time_of_day_ns(time: NmeaTime) -> u64 {
    ((u64::from(time.hour) * 3600 + u64::from(time.minute) * 60 + u64::from(time.second))
        * 1_000_000_000)
        + u64::from(time.nanos)
}

fn next_line_end(buffer: &[u8]) -> Option<(usize, usize)> {
    let pos = buffer.iter().position(|&b| b == b'\n' || b == b'\r')?;
    let term_len = if buffer[pos] == b'\r' && buffer.get(pos + 1) == Some(&b'\n') {
        2
    } else {
        1
    };
    Some((pos, term_len))
}

fn push_line(
    accumulator: &mut NmeaAccumulator,
    line: &[u8],
    line_number: usize,
    output: &mut NmeaChunkOutput,
) {
    if line.is_empty() {
        return;
    }
    let parsed = match std::str::from_utf8(line) {
        Ok(line) => super::parse_sentence(line),
        Err(_) => {
            super::push_error_skip_at(
                &mut output.diagnostics,
                super::NmeaError::NotFramed {
                    reason: "non-ASCII byte",
                },
                RecordRef::at_line(line_number),
            );
            return;
        }
    };
    match parsed {
        Ok(Parsed {
            value,
            mut diagnostics,
        }) => {
            super::set_diagnostic_lines(&mut diagnostics, line_number);
            super::merge_diagnostics(&mut output.diagnostics, diagnostics);
            if let Some(snapshot) = accumulator.push(&value) {
                output.snapshots.push(snapshot);
            }
            output.sentences.push(value);
        }
        Err(error) => super::push_error_skip_at(
            &mut output.diagnostics,
            error,
            RecordRef::at_line(line_number),
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Per-chunk results returned by [`NmeaAccumulator::push_bytes`].
/// Completed snapshots and parsed sentences are kept separate from parser diagnostics, while epoch-assembly warnings remain on each snapshot.
pub struct NmeaChunkOutput {
    /// Snapshots completed while complete lines in the chunk were processed, in processing order; the still-open epoch is omitted.
    pub snapshots: Vec<EpochSnapshot>,
    /// Successfully parsed sentences from complete lines in input order; skipped or still-buffered lines are omitted.
    pub sentences: Vec<NmeaSentence>,
    /// Parser skips and warnings with one-based line references, including the retained-line length-cap skip.
    pub diagnostics: Diagnostics,
}
