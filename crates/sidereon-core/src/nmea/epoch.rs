use std::collections::BTreeSet;

use super::{
    Gga, Gll, Gsa, Gst, Gsv, GsvSatellite, NmeaBody, NmeaDate, NmeaSatNumber, NmeaSentence,
    NmeaSignalId, NmeaTalker, NmeaTime, Rmc, Vtg, Zda,
};
use crate::format::{Diagnostics, Parsed, RecordRef, Warning, WarningKind};

const RETAINED_CAP: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct EpochSnapshot {
    pub time_of_day: Option<NmeaTime>,
    pub date: Option<NmeaDate>,
    pub gga: Option<Gga>,
    pub rmc: Option<Rmc>,
    pub gll: Option<Gll>,
    pub gst: Option<Gst>,
    pub vtg: Option<Vtg>,
    pub zda: Option<Zda>,
    pub gsa: Vec<GsaEntry>,
    pub gsv: Vec<GsvGroup>,
    pub sentence_count: usize,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GsaEntry {
    pub system: Option<crate::GnssSystem>,
    pub gsa: Gsa,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GsvGroup {
    pub talker: NmeaTalker,
    pub signal: Option<NmeaSignalId>,
    pub claimed_in_view: Option<u16>,
    pub satellites: Vec<GsvSatellite>,
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

    pub fn pdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.pdop)
    }

    pub fn hdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.hdop)
    }

    pub fn vdop(&self) -> Option<f64> {
        self.gsa.iter().find_map(|entry| entry.gsa.vdop)
    }

    pub fn used_satellites(&self) -> impl Iterator<Item = &NmeaSatNumber> {
        self.gsa
            .iter()
            .flat_map(|entry| entry.gsa.satellites.iter())
    }

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

    pub fn with_date(date: NmeaDate) -> Self {
        Self {
            carried_date: Some(date),
            ..Self::new()
        }
    }

    pub fn with_max_sentences_per_epoch(mut self, max: usize) -> Self {
        self.max_sentences_per_epoch = max.max(16);
        self
    }

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

    pub fn finish(&mut self) -> Option<EpochSnapshot> {
        if !self.retained.is_empty() {
            let line = std::mem::take(&mut self.retained);
            let mut output = NmeaChunkOutput::default();
            push_line(self, &line, self.next_line, &mut output);
            self.next_line += 1;
        }
        self.current.take().map(|epoch| epoch.snapshot)
    }

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
pub struct NmeaChunkOutput {
    pub snapshots: Vec<EpochSnapshot>,
    pub sentences: Vec<NmeaSentence>,
    pub diagnostics: Diagnostics,
}
