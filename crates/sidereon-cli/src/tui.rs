use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Once;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{bail, Context, Result};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Sparkline, Table};
use ratatui::{Frame, Terminal};

use sidereon::astro::time::civil_from_j2000_seconds;
use sidereon::constants::{
    BDS_EPOCH_MINUS_GPS_EPOCH_S, F_L1_HZ, GPST_MINUS_BDT_S, GPS_EPOCH_TO_J2000_S, SECONDS_PER_WEEK,
};
use sidereon::ephemeris::BroadcastEphemeris;
use sidereon::observables::{predict, ObservableEphemerisSource, PredictOptions};
use sidereon::positioning::{RinexSppOptions, SolveInputs, SolvePolicy};
use sidereon::rinex::observations::ObsEpochTime;
use sidereon::rinex::observations::SignalPolicy;
use sidereon::rtcm::{self, Message, MsmMessage, SsrStreamAssembler};
use sidereon::GnssSystem;

use sidereon::{metrics_from_position_covariance, vertical_radius_at};
use sidereon_core::ntrip::{
    GgaPosition, NtripClientMachine, NtripConfig, NtripCredentials, NtripEvent, NtripVersion,
};
use sidereon_core::positioning::ReceiverSolution;

use crate::{format_epoch, rad_to_deg};

const TICK_RATE: Duration = Duration::from_millis(50);
const MAX_SPEED: f64 = 1024.0;
const MIN_SPEED: f64 = 0.25;
const CONVERGENCE_SAMPLES: usize = 48;
const MAX_ADVANCE_FRAMES_PER_TICK: usize = 8;
const READ_CHUNK: usize = 32 * 1024;
const NTRIP_GGA_INTERVAL_S: f64 = 10.0;
const LIVE_READ_TIMEOUT: Duration = Duration::from_millis(200);

#[derive(Clone, Debug)]
pub(crate) struct NtripConfigInput {
    pub host: String,
    pub port: u16,
    pub mount: String,
    pub user: String,
    pub pass: String,
    pub gga_lat: Option<f64>,
    pub gga_lon: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct TcpConfigInput {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug)]
enum LiveSourceConfig {
    Ntrip(NtripConfigInput),
    Tcp(TcpConfigInput),
}

#[derive(Clone, Debug)]
pub(crate) enum LiveMode {
    Replay,
    Ntrip(NtripConfigInput),
    Tcp(TcpConfigInput),
}

pub(crate) fn run_tui(
    obs_path: Option<&Path>,
    nav_path: &Path,
    speed: f64,
    paused: bool,
    mode: LiveMode,
) -> Result<()> {
    validate_speed(speed)?;
    let mut driver = match mode {
        LiveMode::Replay => {
            let obs = obs_path.context("replay mode requires --obs")?;
            TuiDriver::Replay(Box::new(ReplayDriver::from_files(
                obs, nav_path, speed, paused,
            )?))
        }
        LiveMode::Ntrip(config) => TuiDriver::Live(Box::new(LiveDriver::from_ntrip(
            nav_path, speed, paused, config,
        )?)),
        LiveMode::Tcp(config) => TuiDriver::Live(Box::new(LiveDriver::from_tcp(
            nav_path, speed, paused, config,
        )?)),
    };

    let obs_label = obs_path
        .map(compact_path)
        .unwrap_or_else(|| "live".to_string());
    let nav_label = compact_path(nav_path);

    let mut state = TuiState::new(
        &obs_label,
        &nav_label,
        driver.total_epochs(),
        driver.speed(),
        driver.is_paused(),
    );
    state.connection_status = driver.status_text();

    if let Some(frame) = driver.step_forward()? {
        state.apply_frame(&frame);
        state.connection_status = driver.status_text();
    }

    let mut terminal = TerminalSession::enter()?;
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|frame| render(frame, &state))?;

        let timeout = TICK_RATE
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::ZERO);
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(' ') => {
                        driver.toggle_pause();
                        state.set_paused(driver.is_paused());
                        state.connection_status = driver.status_text();
                        last_tick = Instant::now();
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        if let TuiDriver::Replay(driver) = &mut driver {
                            driver.speed_up();
                        }
                        state.set_speed(driver.speed());
                        state.connection_status = driver.status_text();
                    }
                    KeyCode::Char('-') => {
                        if let TuiDriver::Replay(driver) = &mut driver {
                            driver.speed_down();
                        }
                        state.set_speed(driver.speed());
                        state.connection_status = driver.status_text();
                    }
                    KeyCode::Right | KeyCode::Down if driver.is_paused() => {
                        if let Some(frame) = driver.step_forward()? {
                            state.apply_frame(&frame);
                            state.connection_status = driver.status_text();
                        }
                    }
                    KeyCode::Left | KeyCode::Up if driver.is_paused() => {
                        if let Some(frame) = driver.step_backward()? {
                            state.apply_frame(&frame);
                            state.connection_status = driver.status_text();
                        }
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= TICK_RATE {
            let elapsed = last_tick.elapsed();
            last_tick = Instant::now();
            for frame in driver.advance(elapsed)? {
                state.apply_frame(&frame);
                state.connection_status = driver.status_text();
            }
        }
    }
    Ok(())
}

enum TuiDriver {
    Replay(Box<ReplayDriver>),
    Live(Box<LiveDriver>),
}

impl TuiDriver {
    fn step_forward(&mut self) -> Result<Option<MonitorFrame>> {
        match self {
            Self::Replay(driver) => driver.step_forward(),
            Self::Live(driver) => driver.poll_live().map(|frames| frames.into_iter().next()),
        }
    }

    fn step_backward(&mut self) -> Result<Option<MonitorFrame>> {
        match self {
            Self::Replay(driver) => driver.step_backward(),
            Self::Live(_driver) => Ok(None),
        }
    }

    fn advance(&mut self, wall_delta: Duration) -> Result<Vec<MonitorFrame>> {
        match self {
            Self::Replay(driver) => driver.advance(wall_delta),
            Self::Live(driver) => {
                if wall_delta.is_zero() {
                    Ok(Vec::new())
                } else {
                    driver.poll_live()
                }
            }
        }
    }

    fn speed(&self) -> f64 {
        match self {
            Self::Replay(driver) => driver.speed(),
            Self::Live(driver) => driver.speed(),
        }
    }

    fn is_paused(&self) -> bool {
        match self {
            Self::Replay(driver) => driver.is_paused(),
            Self::Live(driver) => driver.is_paused(),
        }
    }

    fn toggle_pause(&mut self) {
        match self {
            Self::Replay(driver) => driver.toggle_pause(),
            Self::Live(driver) => driver.toggle_pause(),
        }
    }

    fn total_epochs(&self) -> usize {
        match self {
            Self::Replay(driver) => driver.len(),
            Self::Live(driver) => driver.len(),
        }
    }

    fn status_text(&self) -> String {
        match self {
            Self::Replay(driver) => driver.status_text(),
            Self::Live(driver) => driver.status_text(),
        }
    }
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        install_panic_hook();
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide) {
            let _ = disable_raw_mode();
            return Err(error).context("enter alternate screen");
        }
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                restore_terminal_for_panic();
                return Err(error).context("create terminal");
            }
        };
        if let Err(error) = terminal.clear() {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen, Show);
            return Err(error).context("clear terminal");
        }
        Ok(Self { terminal })
    }

    fn draw<F>(&mut self, draw: F) -> io::Result<()>
    where
        F: FnOnce(&mut Frame),
    {
        self.terminal.draw(draw).map(|_| ())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen, Show);
        let _ = self.terminal.show_cursor();
    }
}

fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal_for_panic();
            previous(info);
        }));
    });
}

fn restore_terminal_for_panic() {
    let _ = disable_raw_mode();
    let mut stdout = io::stdout();
    let _ = execute!(stdout, LeaveAlternateScreen, Show);
}

#[derive(Debug)]
struct RtcmEpochMapper {
    raw_anchor: BTreeMap<GnssSystem, RawState>,
    source_weeks: BTreeMap<GnssSystem, Vec<u32>>,
}

#[derive(Clone, Debug)]
struct RawState {
    last_raw: Option<u32>,
    last_continuous_ms: Option<f64>,
    anchor_week: Option<u32>,
}

impl RtcmEpochMapper {
    fn new(source: &BroadcastEphemeris) -> Self {
        let mut source_weeks = BTreeMap::new();
        for record in source.records() {
            let entry = source_weeks
                .entry(record.satellite_id.system)
                .or_insert(Vec::new());
            if !entry.contains(&record.week) {
                entry.push(record.week);
            }
        }
        for weeks in source_weeks.values_mut() {
            weeks.sort_unstable();
        }

        let mut raw_anchor = BTreeMap::new();
        for (system, rec_week) in source
            .records()
            .iter()
            .map(|record| (record.satellite_id.system, record.week))
        {
            raw_anchor.entry(system).or_insert(RawState {
                last_raw: None,
                last_continuous_ms: None,
                anchor_week: Some(rec_week),
            });
        }

        Self {
            raw_anchor,
            source_weeks,
        }
    }

    fn map_epoch(
        &mut self,
        system: GnssSystem,
        raw_epoch_time: u32,
    ) -> Option<(f64, ObsEpochTime)> {
        let state = self.raw_anchor.entry(system).or_insert(RawState {
            last_raw: None,
            last_continuous_ms: None,
            anchor_week: None,
        });

        let continuous_ms =
            if let (Some(last_raw), Some(last_ms)) = (state.last_raw, state.last_continuous_ms) {
                let delta = rtcm::msm_epoch_dt_ms(system, last_raw, raw_epoch_time) as f64;
                last_ms + delta
            } else {
                let base_week = state.anchor_week.or_else(|| {
                    self.source_weeks
                        .get(&system)
                        .and_then(|weeks| weeks.first().copied())
                })?;
                state.anchor_week = Some(base_week);
                let base_ms = if system == GnssSystem::Glonass {
                    let day = raw_epoch_time >> 27;
                    let ms = raw_epoch_time & ((1_u32 << 27) - 1);
                    f64::from(day * 86_400_000u32 + ms)
                } else {
                    f64::from(raw_epoch_time)
                };
                f64::from(base_week) * SECONDS_PER_WEEK * 1000.0 + base_ms
            };

        state.last_raw = Some(raw_epoch_time);
        state.last_continuous_ms = Some(continuous_ms);

        let t_rx_j2000_s = match system {
            GnssSystem::BeiDou => {
                GPS_EPOCH_TO_J2000_S
                    + GPST_MINUS_BDT_S
                    + BDS_EPOCH_MINUS_GPS_EPOCH_S
                    + continuous_ms / 1000.0
            }
            _ => GPS_EPOCH_TO_J2000_S + continuous_ms / 1000.0,
        };

        let split = split_to_civil(t_rx_j2000_s);
        Some((t_rx_j2000_s, split))
    }
}

fn split_to_civil(t_rx_j2000_s: f64) -> ObsEpochTime {
    let base = civil_from_j2000_seconds(t_rx_j2000_s.floor() as i64);
    let second_of_day = split_seconds_of_day(t_rx_j2000_s);
    let minute = (second_of_day % 3600) / 60;
    let second = (second_of_day % 60) as f64 + (t_rx_j2000_s.fract().abs());
    ObsEpochTime {
        year: base.0 as i32,
        month: u8::try_from(base.1).expect("month to u8"),
        day: u8::try_from(base.2).expect("day to u8"),
        hour: u8::try_from(base.3).expect("hour to u8"),
        minute: u8::try_from(minute).expect("minute to u8"),
        second,
    }
}

fn split_seconds_of_day(t_rx_j2000_s: f64) -> u32 {
    let floored = t_rx_j2000_s.floor() as i64;
    let second_of_day = (floored % 86_400 + 86_400) % 86_400;
    u32::try_from(second_of_day).expect("seconds in day")
}

struct ReplayDriver {
    nav: BroadcastEphemeris,
    epochs: Vec<sidereon::positioning::RinexSppEpochInputs>,
    timeline: ReplayTimeline,
}

impl ReplayDriver {
    fn from_files(obs_path: &Path, nav_path: &Path, speed: f64, paused: bool) -> Result<Self> {
        validate_speed(speed)?;
        let obs = sidereon::load_rinex_obs(obs_path)
            .with_context(|| format!("load OBS {}", obs_path.display()))?;
        let nav = sidereon::load_rinex_nav(nav_path)
            .with_context(|| format!("load NAV {}", nav_path.display()))?;
        let options =
            RinexSppOptions::default_for(&obs).context("build default RINEX SPP options")?;
        let epochs = sidereon::spp_inputs_from_rinex_obs(&obs, &nav, &options)
            .context("assemble RINEX SPP inputs")?;
        if epochs.is_empty() {
            bail!("RINEX OBS has no replayable SPP epochs");
        }
        let epoch_times_s = epochs
            .iter()
            .map(|epoch| epoch.inputs.t_rx_j2000_s)
            .collect();
        Ok(Self {
            nav,
            epochs,
            timeline: ReplayTimeline::new(epoch_times_s, speed, paused)?,
        })
    }

    fn len(&self) -> usize {
        self.epochs.len()
    }

    fn speed(&self) -> f64 {
        self.timeline.speed()
    }

    fn is_paused(&self) -> bool {
        self.timeline.is_paused()
    }

    fn toggle_pause(&mut self) {
        self.timeline.set_paused(!self.timeline.is_paused())
    }

    fn speed_up(&mut self) {
        self.timeline.speed_up()
    }

    fn speed_down(&mut self) {
        self.timeline.speed_down()
    }

    fn step_forward(&mut self) -> Result<Option<MonitorFrame>> {
        self.timeline
            .step_forward()
            .map(|index| self.frame_at(index))
            .transpose()
    }

    fn step_backward(&mut self) -> Result<Option<MonitorFrame>> {
        self.timeline
            .step_backward()
            .map(|index| self.frame_at(index))
            .transpose()
    }

    fn advance(&mut self, wall_delta: Duration) -> Result<Vec<MonitorFrame>> {
        self.timeline
            .advance_wall_time(wall_delta)
            .into_iter()
            .map(|index| self.frame_at(index))
            .collect()
    }

    fn frame_at(&self, replay_index: usize) -> Result<MonitorFrame> {
        let epoch = self
            .epochs
            .get(replay_index)
            .with_context(|| format!("replay index {replay_index} out of range"))?;
        let solution = sidereon::solve_spp_batch_serial(
            &self.nav,
            std::slice::from_ref(&epoch.inputs),
            true,
            SolvePolicy::default(),
        )
        .into_iter()
        .next()
        .context("missing SPP solve result")?;
        let satellites = satellite_snapshots(&self.nav, &epoch.inputs, solution.as_ref().ok());
        Ok(MonitorFrame::Replay(ReplayFrame {
            replay_index,
            raw_epoch: Some(epoch.epoch_index),
            epoch: epoch.epoch,
            observation_count: epoch.inputs.observations.len(),
            solution,
            satellites,
        }))
    }

    fn status_text(&self) -> String {
        if self.is_paused() {
            "paused".to_string()
        } else {
            "replay".to_string()
        }
    }
}

struct LiveDriver {
    nav: BroadcastEphemeris,
    options: RinexSppOptions,
    timeline_speed: f64,
    paused: bool,
    status: String,
    source: LiveSourceConfig,
    stream: Option<TcpStream>,
    ntrip_machine: Option<NtripClientMachine>,
    assembler: SsrStreamAssembler,
    mapper: RtcmEpochMapper,
    epoch_buffer: Vec<MsmMessage>,
    reconnect_attempts: usize,
    pending_reconnect_at: Option<SystemTime>,
    total_frames: usize,
    gga_lat: Option<f64>,
    gga_lon: Option<f64>,
    connect_started: Option<SystemTime>,
}

impl LiveDriver {
    fn from_ntrip(
        nav_path: &Path,
        speed: f64,
        paused: bool,
        input: NtripConfigInput,
    ) -> Result<Self> {
        let nav = sidereon::load_rinex_nav(nav_path)?;
        let mapper = RtcmEpochMapper::new(&nav);
        let gga_lat = input.gga_lat;
        let gga_lon = input.gga_lon;
        let host = input.host.clone();
        let port = input.port;
        let mount = input.mount.clone();
        let user = input.user.clone();
        let pass = input.pass.clone();
        let options =
            RinexSppOptions::new(SignalPolicy::default_for(3.05)?).with_initial_guess([0.0; 4]);
        let credentials = if user.is_empty() || pass.is_empty() {
            None
        } else {
            Some(NtripCredentials {
                username: user,
                password: pass,
            })
        };
        let mut ntrip_config = NtripConfig::default();
        ntrip_config.host = host;
        ntrip_config.port = port;
        ntrip_config.mountpoint = mount;
        ntrip_config.version = NtripVersion::Rev2;
        ntrip_config.credentials = credentials;
        ntrip_config.user_agent_product = format!("sidereon/{}", env!("CARGO_PKG_VERSION"));
        ntrip_config.gga_interval_s = Some(NTRIP_GGA_INTERVAL_S);
        let mut machine = NtripClientMachine::new(ntrip_config);
        machine.reset();
        Ok(Self {
            nav,
            options,
            timeline_speed: speed,
            paused,
            status: "disconnected".to_string(),
            source: LiveSourceConfig::Ntrip(input),
            stream: None,
            ntrip_machine: Some(machine),
            assembler: SsrStreamAssembler::new(),
            mapper,
            epoch_buffer: Vec::new(),
            reconnect_attempts: 0,
            pending_reconnect_at: None,
            total_frames: 0,
            gga_lat,
            gga_lon,
            connect_started: None,
        })
    }

    fn from_tcp(nav_path: &Path, speed: f64, paused: bool, input: TcpConfigInput) -> Result<Self> {
        let nav = sidereon::load_rinex_nav(nav_path)?;
        let mapper = RtcmEpochMapper::new(&nav);
        let options =
            RinexSppOptions::new(SignalPolicy::default_for(3.05)?).with_initial_guess([0.0; 4]);
        Ok(Self {
            nav,
            options,
            timeline_speed: speed,
            paused,
            status: "disconnected".to_string(),
            source: LiveSourceConfig::Tcp(input),
            stream: None,
            ntrip_machine: None,
            assembler: SsrStreamAssembler::new(),
            mapper,
            epoch_buffer: Vec::new(),
            reconnect_attempts: 0,
            pending_reconnect_at: None,
            total_frames: 0,
            gga_lat: None,
            gga_lon: None,
            connect_started: None,
        })
    }

    fn len(&self) -> usize {
        self.total_frames
    }

    fn speed(&self) -> f64 {
        self.timeline_speed
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    fn status_text(&self) -> String {
        self.status.clone()
    }

    fn maybe_reconnect(&mut self) -> Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        if let Some(at) = self.pending_reconnect_at {
            if SystemTime::now() < at {
                return Ok(());
            }
        }
        self.connect()
    }

    fn schedule_reconnect(&mut self, reason: impl std::fmt::Display) {
        self.connect_started = None;
        self.stream = None;
        self.reconnect_attempts = self.reconnect_attempts.saturating_add(1);
        let mut delay = Duration::from_millis(500);
        for _ in 0..self.reconnect_attempts.min(6) {
            delay = delay.saturating_mul(2);
        }
        self.pending_reconnect_at = Some(SystemTime::now() + delay);
        self.status = format!("reconnecting: {}", reason);
    }

    fn connect(&mut self) -> Result<()> {
        let (host, port) = match &self.source {
            LiveSourceConfig::Ntrip(config) => (config.host.as_str(), config.port),
            LiveSourceConfig::Tcp(config) => (config.host.as_str(), config.port),
        };
        let mut stream = TcpStream::connect(format!("{host}:{port}"))
            .with_context(|| format!("connect to {host}:{port}"))?;
        stream.set_read_timeout(Some(LIVE_READ_TIMEOUT))?;
        match &self.source {
            LiveSourceConfig::Ntrip(config) => {
                let Some(machine) = &mut self.ntrip_machine else {
                    bail!("missing ntrip machine");
                };
                let request = machine
                    .connection_request()
                    .context("build ntrip request")?;
                stream.write_all(&request).context("send ntrip request")?;
                self.gga_lat = config.gga_lat;
                self.gga_lon = config.gga_lon;
                self.status = "ntrip connecting".to_string();
            }
            LiveSourceConfig::Tcp(_) => {
                self.status = "tcp connected".to_string();
            }
        }
        self.connect_started = Some(SystemTime::now());
        self.stream = Some(stream);
        self.reconnect_attempts = 0;
        self.pending_reconnect_at = None;
        Ok(())
    }

    fn poll_live(&mut self) -> Result<Vec<MonitorFrame>> {
        if self.paused {
            return Ok(Vec::new());
        }
        if self.stream.is_none() {
            self.maybe_reconnect()?;
        }
        if self.stream.is_none() {
            return Ok(Vec::new());
        }

        let mut chunk = vec![0u8; READ_CHUNK];
        let read = match self
            .stream
            .as_mut()
            .context("missing stream")?
            .read(&mut chunk)
        {
            Ok(0) => {
                self.schedule_reconnect("stream ended");
                return Ok(Vec::new());
            }
            Ok(size) => size,
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Ok(Vec::new());
            }
            Err(error) => {
                self.schedule_reconnect(error.to_string());
                return Ok(Vec::new());
            }
        };
        chunk.truncate(read);
        let events = match &self.source {
            LiveSourceConfig::Ntrip(_) => self.process_ntrip_bytes(&chunk)?,
            LiveSourceConfig::Tcp(_) => vec![chunk],
        };

        let mut frames = Vec::new();
        for payload in events {
            for parsed in self.assembler.push(&payload) {
                match parsed {
                    Ok(Message::Msm(message)) => self.epoch_buffer.push(message),
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }

        if self.epoch_buffer.is_empty() {
            return Ok(Vec::new());
        }

        let messages = core::mem::take(&mut self.epoch_buffer);
        let solved = self.solve_rtcm_epoch_messages(messages)?;
        frames.extend(solved);

        Ok(frames)
    }

    fn process_ntrip_bytes(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>> {
        let Some(machine) = &mut self.ntrip_machine else {
            bail!("missing ntrip machine");
        };
        let mut payloads = Vec::new();
        for event in machine.push(chunk) {
            match event {
                NtripEvent::Connected(_) => {
                    self.status = "streaming".to_string();
                }
                NtripEvent::Payload(payload) => payloads.push(payload),
                NtripEvent::Sourcetable(_) => {
                    self.status = "sourcetable".to_string();
                }
                NtripEvent::StreamEnded | NtripEvent::StreamCorrupted { .. } => {
                    self.schedule_reconnect("stream corrupted");
                }
                NtripEvent::Rejected(rejection) => {
                    self.schedule_reconnect(format!("rejected: {rejection:?}"));
                }
            }
        }
        Ok(payloads)
    }

    fn solve_rtcm_epoch_messages(
        &mut self,
        messages: Vec<MsmMessage>,
    ) -> Result<Vec<MonitorFrame>> {
        let epochs = sidereon::spp_inputs_from_rtcm_msm(
            &messages,
            &self.nav,
            &self.options,
            |system, raw| self.mapper.map_epoch(system, raw),
        )
        .context("convert RTCM MSM stream")?;

        let mut frames = Vec::new();
        for epoch in epochs {
            let mut solved = sidereon::solve_spp_batch_serial(
                &self.nav,
                std::slice::from_ref(&epoch.inputs),
                true,
                SolvePolicy::default(),
            )
            .into_iter();
            let solve_result = solved.next().context("missing solve result")?;

            self.total_frames = self.total_frames.saturating_add(1);

            self.send_gga(&solve_result)?;

            let satellites =
                satellite_snapshots(&self.nav, &epoch.inputs, solve_result.as_ref().ok());
            frames.push(MonitorFrame::Replay(ReplayFrame {
                replay_index: epoch.epoch_index,
                raw_epoch: Some(epoch.epoch_index),
                epoch: epoch.epoch,
                observation_count: epoch.inputs.observations.len(),
                solution: solve_result,
                satellites,
            }));
        }

        Ok(frames)
    }

    fn send_gga(&mut self, solution: &sidereon::Result<ReceiverSolution>) -> Result<()> {
        let (mut lat_deg, mut lon_deg, height_m) = match solution
            .as_ref()
            .ok()
            .and_then(|solution| solution.geodetic)
        {
            Some(geo) => (
                geo.lat_rad.to_degrees(),
                geo.lon_rad.to_degrees(),
                geo.height_m,
            ),
            None => return Ok(()),
        };
        if let Some(lat) = self.gga_lat {
            lat_deg = lat;
        }
        if let Some(lon) = self.gga_lon {
            lon_deg = lon;
        }

        let Some(stream) = self.stream.as_mut() else {
            return Ok(());
        };
        let Some(machine) = self.ntrip_machine.as_mut() else {
            return Ok(());
        };
        let Some(started) = self.connect_started else {
            return Ok(());
        };
        let Some(now) = started.elapsed().ok().map(|elapsed| elapsed.as_secs_f64()) else {
            return Ok(());
        };

        let position = GgaPosition {
            lat_deg,
            lon_deg,
            height_m,
            fix_quality: 1,
            num_satellites: 10,
            hdop: 1.0,
        };
        if let Some(message) = machine.gga_message(now, &position, now.rem_euclid(86_400.0)) {
            stream.write_all(&message)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn poll_from_reader<R: Read>(&mut self, reader: &mut R) -> Result<Vec<MonitorFrame>> {
        let mut chunk = vec![0u8; READ_CHUNK];
        let read = match reader.read(&mut chunk) {
            Ok(0) => return Ok(Vec::new()),
            Ok(size) => size,
            Err(error) => {
                self.schedule_reconnect(error.to_string());
                return Ok(Vec::new());
            }
        };
        chunk.truncate(read);
        let events = match &self.source {
            LiveSourceConfig::Ntrip(_) => self.process_ntrip_bytes(&chunk)?,
            LiveSourceConfig::Tcp(_) => vec![chunk],
        };

        for payload in events {
            for parsed in self.assembler.push(&payload) {
                if let Ok(Message::Msm(message)) = parsed {
                    self.epoch_buffer.push(message)
                }
            }
        }
        let messages = core::mem::take(&mut self.epoch_buffer);
        self.solve_rtcm_epoch_messages(messages)
    }
}

#[derive(Debug)]
enum MonitorFrame {
    Replay(ReplayFrame),
}

#[derive(Debug)]
struct ReplayFrame {
    replay_index: usize,
    raw_epoch: Option<usize>,
    epoch: ObsEpochTime,
    observation_count: usize,
    solution: sidereon::Result<ReceiverSolution>,
    satellites: Vec<SatelliteSnapshot>,
}

fn validate_speed(speed: f64) -> Result<()> {
    if !speed.is_finite() || speed <= 0.0 {
        bail!("--speed must be a finite positive multiplier");
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct ReplayTimeline {
    epoch_times_s: Vec<f64>,
    current_index: Option<usize>,
    accumulated_replay_s: f64,
    speed: f64,
    paused: bool,
}

impl ReplayTimeline {
    fn new(epoch_times_s: Vec<f64>, speed: f64, paused: bool) -> Result<Self> {
        validate_speed(speed)?;
        if epoch_times_s.is_empty() {
            bail!("replay timeline requires at least one epoch");
        }
        if epoch_times_s.iter().any(|value| !value.is_finite()) {
            bail!("replay timeline contains non-finite epoch time");
        }
        Ok(Self {
            epoch_times_s,
            current_index: None,
            accumulated_replay_s: 0.0,
            speed,
            paused,
        })
    }

    fn speed(&self) -> f64 {
        self.speed
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        self.accumulated_replay_s = 0.0;
    }

    fn speed_up(&mut self) {
        self.speed = (self.speed * 2.0).min(MAX_SPEED);
    }

    fn speed_down(&mut self) {
        self.speed = (self.speed / 2.0).max(MIN_SPEED);
    }

    fn step_forward(&mut self) -> Option<usize> {
        let next = self.current_index.map_or(0, |index| index + 1);
        if next >= self.epoch_times_s.len() {
            return None;
        }
        self.current_index = Some(next);
        self.accumulated_replay_s = 0.0;
        Some(next)
    }

    fn step_backward(&mut self) -> Option<usize> {
        let current = self.current_index?;
        let previous = current.checked_sub(1)?;
        self.current_index = Some(previous);
        self.accumulated_replay_s = 0.0;
        Some(previous)
    }

    fn advance_wall_time(&mut self, wall_delta: Duration) -> Vec<usize> {
        if self.paused {
            return Vec::new();
        }
        let mut emitted = Vec::new();
        if self.current_index.is_none() {
            self.current_index = Some(0);
            emitted.push(0);
        }
        self.accumulated_replay_s += wall_delta.as_secs_f64() * self.speed;

        while emitted.len() < MAX_ADVANCE_FRAMES_PER_TICK {
            let Some(current) = self.current_index else {
                break;
            };
            let next = current + 1;
            if next >= self.epoch_times_s.len() {
                break;
            }
            let gap_s = (self.epoch_times_s[next] - self.epoch_times_s[current]).max(0.0);
            if gap_s > self.accumulated_replay_s {
                break;
            }
            self.accumulated_replay_s -= gap_s;
            self.current_index = Some(next);
            emitted.push(next);
        }
        emitted
    }
}

#[derive(Debug)]
struct TuiState {
    obs_label: String,
    nav_label: String,
    total_epochs: usize,
    current_replay_epoch: Option<usize>,
    current_raw_epoch: Option<usize>,
    observation_count: usize,
    epoch_time: String,
    status: String,
    connection_status: String,
    speed: f64,
    paused: bool,
    lat_deg: Option<f64>,
    lon_deg: Option<f64>,
    height_m: Option<f64>,
    bounds: ErrorBounds,
    satellites: Vec<SatelliteSnapshot>,
    origin: Option<ConvergenceOrigin>,
    latest_horizontal_m: Option<f64>,
    convergence_m: VecDeque<f64>,
}

impl TuiState {
    fn new(
        obs_label: &str,
        nav_label: &str,
        total_epochs: usize,
        speed: f64,
        paused: bool,
    ) -> Self {
        Self {
            obs_label: obs_label.to_string(),
            nav_label: nav_label.to_string(),
            total_epochs,
            current_replay_epoch: None,
            current_raw_epoch: None,
            observation_count: 0,
            epoch_time: "n/a".to_string(),
            status: "ready".to_string(),
            connection_status: "ready".to_string(),
            speed,
            paused,
            lat_deg: None,
            lon_deg: None,
            height_m: None,
            bounds: ErrorBounds::empty(),
            satellites: Vec::new(),
            origin: None,
            latest_horizontal_m: None,
            convergence_m: VecDeque::new(),
        }
    }

    fn set_speed(&mut self, speed: f64) {
        self.speed = speed;
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    fn apply_frame(&mut self, frame: &MonitorFrame) {
        match frame {
            MonitorFrame::Replay(frame) => {
                self.current_replay_epoch = Some(frame.replay_index + 1);
                self.current_raw_epoch = frame.raw_epoch;
                self.observation_count = frame.observation_count;
                self.epoch_time = format_epoch(frame.epoch);
                self.satellites = frame.satellites.clone();
                match &frame.solution {
                    Ok(solution) => {
                        self.status = if solution.metadata.converged {
                            "solved".to_string()
                        } else {
                            "not converged".to_string()
                        };
                        self.lat_deg = solution.geodetic.map(|geo| rad_to_deg(geo.lat_rad));
                        self.lon_deg = solution.geodetic.map(|geo| rad_to_deg(geo.lon_rad));
                        self.height_m = solution.geodetic.map(|geo| geo.height_m);
                        self.bounds = ErrorBounds::from_solution(solution);
                        self.push_convergence(solution);
                    }
                    Err(error) => {
                        self.status = format!("error: {error}");
                        self.lat_deg = None;
                        self.lon_deg = None;
                        self.height_m = None;
                        self.bounds = ErrorBounds::empty();
                    }
                }
            }
        }
    }

    fn push_convergence(&mut self, solution: &ReceiverSolution) {
        let ecef = solution.position.as_array();
        if self.origin.is_none() {
            if let Some(geo) = solution.geodetic {
                self.origin = Some(ConvergenceOrigin { ecef, geo });
            }
        }
        let Some(origin) = self.origin else {
            return;
        };
        let delta = [
            ecef[0] - origin.ecef[0],
            ecef[1] - origin.ecef[1],
            ecef[2] - origin.ecef[2],
        ];
        let enu = sidereon::dop::ecef_to_enu_rotation(origin.geo.lat_rad, origin.geo.lon_rad);
        let east_m = dot(delta, enu[0]);
        let north_m = dot(delta, enu[1]);
        let horizontal_m = libm::hypot(east_m, north_m);
        self.latest_horizontal_m = Some(horizontal_m);
        if self.convergence_m.len() == CONVERGENCE_SAMPLES {
            self.convergence_m.pop_front();
        }
        self.convergence_m.push_back(horizontal_m);
    }
}

#[derive(Clone, Copy, Debug)]
struct ConvergenceOrigin {
    ecef: [f64; 3],
    geo: sidereon::Wgs84Geodetic,
}

#[derive(Debug, Clone)]
struct ErrorBounds {
    cep_m: Option<f64>,
    r95_m: Option<f64>,
    vertical_95_m: Option<f64>,
    sigma_e_m: Option<f64>,
    sigma_n_m: Option<f64>,
    sigma_u_m: Option<f64>,
}

impl ErrorBounds {
    fn empty() -> Self {
        Self {
            cep_m: None,
            r95_m: None,
            vertical_95_m: None,
            sigma_e_m: None,
            sigma_n_m: None,
            sigma_u_m: None,
        }
    }

    fn from_solution(solution: &ReceiverSolution) -> Self {
        let metrics = metrics_from_position_covariance(&solution.position_covariance).ok();
        let vertical_95_m = metrics.as_ref().and_then(|_metrics| {
            vertical_radius_at(solution.position_covariance.enu_m2[2][2], 0.95).ok()
        });
        match (metrics, vertical_95_m) {
            (Some(metrics), Some(vertical_95_m)) => Self {
                cep_m: Some(metrics.cep_m.radius_m),
                r95_m: Some(metrics.r95_m.radius_m),
                vertical_95_m: Some(vertical_95_m),
                sigma_e_m: Some(metrics.sigma_e_m),
                sigma_n_m: Some(metrics.sigma_n_m),
                sigma_u_m: Some(metrics.sigma_u_m),
            },
            _ => Self::empty(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SatelliteSnapshot {
    id: String,
    elevation_deg: Option<f64>,
    azimuth_deg: Option<f64>,
    used: bool,
}

fn satellite_snapshots(
    source: &dyn ObservableEphemerisSource,
    inputs: &SolveInputs,
    solution: Option<&ReceiverSolution>,
) -> Vec<SatelliteSnapshot> {
    let used: BTreeSet<sidereon::GnssSatelliteId> = solution
        .map(|solution| solution.used_sats.iter().copied().collect())
        .unwrap_or_default();
    let receiver_ecef = solution
        .map(|solution| solution.position.as_array())
        .unwrap_or([
            inputs.initial_guess[0],
            inputs.initial_guess[1],
            inputs.initial_guess[2],
        ]);

    inputs
        .observations
        .iter()
        .map(|observation| {
            let (azimuth_deg, elevation_deg) = predict(
                source,
                observation.satellite_id,
                receiver_ecef,
                inputs.t_rx_j2000_s,
                {
                    let mut options = PredictOptions::default();
                    options.carrier_hz = F_L1_HZ;
                    options.light_time = true;
                    options.sagnac = true;
                    options
                },
            )
            .map_or((None, None), |prediction| {
                (Some(prediction.azimuth_deg), Some(prediction.elevation_deg))
            });
            SatelliteSnapshot {
                id: observation.satellite_id.to_string(),
                elevation_deg,
                azimuth_deg,
                used: used.contains(&observation.satellite_id),
            }
        })
        .collect()
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn render(frame: &mut Frame, state: &TuiState) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(12)])
        .split(frame.area());
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(root[0]);
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(root[1]);

    render_solution(frame, top[0], state);
    render_bounds(frame, top[1], state);
    render_satellites(frame, bottom[0], state);
    render_convergence(frame, bottom[1], state);
}

fn render_solution(frame: &mut Frame, area: Rect, state: &TuiState) {
    let epoch = state
        .current_replay_epoch
        .map(|epoch| format!("{epoch}/{}", state.total_epochs))
        .unwrap_or_else(|| format!("0/{}", state.total_epochs));
    let raw_epoch = state
        .current_raw_epoch
        .map(|epoch| epoch.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let lines = vec![
        Line::from(vec![
            Span::styled("time ", label_style()),
            Span::raw(state.epoch_time.clone()),
            Span::raw("   "),
            Span::styled("epoch ", label_style()),
            Span::raw(epoch),
            Span::raw("   "),
            Span::styled("raw ", label_style()),
            Span::raw(raw_epoch),
        ]),
        Line::from(vec![
            Span::styled("status ", label_style()),
            Span::styled(state.status.clone(), status_style(&state.status)),
        ]),
        Line::from(vec![
            Span::styled("conn ", label_style()),
            Span::styled(
                state.connection_status.clone(),
                status_style(&state.connection_status),
            ),
        ]),
        Line::from(vec![
            Span::styled("lat ", label_style()),
            Span::raw(format_optional_deg(state.lat_deg)),
            Span::raw("   "),
            Span::styled("lon ", label_style()),
            Span::raw(format_optional_deg(state.lon_deg)),
            Span::raw("   "),
            Span::styled("height ", label_style()),
            Span::raw(format_optional_m(state.height_m)),
        ]),
        Line::from(vec![
            Span::styled("obs ", label_style()),
            Span::raw(state.obs_label.clone()),
        ]),
        Line::from(vec![
            Span::styled("nav ", label_style()),
            Span::raw(state.nav_label.clone()),
        ]),
        Line::from(vec![
            Span::styled("speed ", label_style()),
            Span::raw(format_speed(state.speed)),
            Span::raw("   "),
            Span::styled("paused ", label_style()),
            Span::raw(if state.paused { "yes" } else { "no" }),
            Span::raw("   "),
            Span::styled("observations ", label_style()),
            Span::raw(state.observation_count.to_string()),
        ]),
    ];
    let paragraph =
        Paragraph::new(lines).block(Block::default().title("Solution").borders(Borders::ALL));
    frame.render_widget(paragraph, area);
}

fn render_bounds(frame: &mut Frame, area: Rect, state: &TuiState) {
    let lines = vec![
        Line::from(vec![
            Span::styled("CEP ", label_style()),
            Span::raw(format_optional_m(state.bounds.cep_m)),
        ]),
        Line::from(vec![
            Span::styled("R95 ", label_style()),
            Span::raw(format_optional_m(state.bounds.r95_m)),
        ]),
        Line::from(vec![
            Span::styled("V95 ", label_style()),
            Span::raw(format_optional_m(state.bounds.vertical_95_m)),
        ]),
        Line::from(vec![
            Span::styled("sigma E ", label_style()),
            Span::raw(format_optional_m(state.bounds.sigma_e_m)),
        ]),
        Line::from(vec![
            Span::styled("sigma N ", label_style()),
            Span::raw(format_optional_m(state.bounds.sigma_n_m)),
        ]),
        Line::from(vec![
            Span::styled("sigma U ", label_style()),
            Span::raw(format_optional_m(state.bounds.sigma_u_m)),
        ]),
    ];
    let paragraph =
        Paragraph::new(lines).block(Block::default().title("Error Bounds").borders(Borders::ALL));
    frame.render_widget(paragraph, area);
}

fn render_satellites(frame: &mut Frame, area: Rect, state: &TuiState) {
    let rows = state.satellites.iter().map(|sat| {
        Row::new(vec![
            Cell::from(sat.id.clone()),
            Cell::from(format_optional_deg(sat.elevation_deg)),
            Cell::from(format_optional_deg(sat.azimuth_deg)),
            Cell::from(if sat.used { "yes" } else { "no" }),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(8),
        ],
    )
    .header(
        Row::new(vec!["sat", "elevation", "azimuth", "used"]).style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(Block::default().title("Satellites").borders(Borders::ALL));
    frame.render_widget(table, area);
}

fn render_convergence(frame: &mut Frame, area: Rect, state: &TuiState) {
    let block = Block::default().title("Convergence").borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(inner);
    let latest = state
        .latest_horizontal_m
        .map(|value| format!("{value:.3} m"))
        .unwrap_or_else(|| "n/a".to_string());
    let sample_count = state.convergence_m.len();
    let paragraph = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("latest horizontal scatter ", label_style()),
            Span::raw(latest),
        ]),
        Line::from(vec![
            Span::styled("samples ", label_style()),
            Span::raw(sample_count.to_string()),
        ]),
    ]);
    frame.render_widget(paragraph, sections[0]);

    let data = convergence_data(&state.convergence_m);
    let sparkline = Sparkline::default()
        .data(&data)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(sparkline, sections[1]);
}

fn label_style() -> Style {
    Style::default()
        .fg(Color::Gray)
        .add_modifier(Modifier::BOLD)
}

fn status_style(status: &str) -> Style {
    if status == "solved" || status == "connected" || status == "streaming" {
        Style::default().fg(Color::Green)
    } else if status.starts_with("error") || status.starts_with("reconnecting") {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Yellow)
    }
}

fn format_optional_deg(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3} deg"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_m(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3} m"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_speed(speed: f64) -> String {
    if (speed.fract()).abs() < f64::EPSILON {
        format!("{speed:.0}x")
    } else {
        format!("{speed:.2}x")
    }
}

fn compact_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn convergence_data(values: &VecDeque<f64>) -> Vec<u64> {
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![0; values.len().max(1)];
    }
    values
        .iter()
        .map(|value| ((*value / max) * 100.0).round().clamp(0.0, 100.0) as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use std::io::{Cursor, Error, ErrorKind, Read};

    fn read_fixture(parts: &[&str]) -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../sidereon-core/tests/fixtures");
        for part in parts {
            path.push(part);
        }
        path
    }

    #[test]
    fn timeline_advances_by_epoch_gap_and_speed() {
        let mut timeline =
            ReplayTimeline::new(vec![0.0, 30.0, 60.0], 10.0, false).expect("timeline");
        assert_eq!(timeline.advance_wall_time(Duration::from_secs(0)), vec![0]);
        assert!(timeline
            .advance_wall_time(Duration::from_millis(2900))
            .is_empty());
        assert_eq!(
            timeline.advance_wall_time(Duration::from_millis(100)),
            vec![1]
        );

        timeline.speed_up();
        assert_eq!(timeline.speed(), 20.0);
        assert!(timeline
            .advance_wall_time(Duration::from_millis(1499))
            .is_empty());
        assert_eq!(
            timeline.advance_wall_time(Duration::from_millis(1)),
            vec![2]
        );
    }

    #[test]
    fn timeline_steps_forward_and_backward_while_paused() {
        let mut timeline =
            ReplayTimeline::new(vec![10.0, 20.0, 30.0], 4.0, true).expect("timeline");
        assert!(timeline
            .advance_wall_time(Duration::from_secs(100))
            .is_empty());
        assert_eq!(timeline.step_forward(), Some(0));
        assert_eq!(timeline.step_forward(), Some(1));
        assert_eq!(timeline.step_backward(), Some(0));
        assert_eq!(timeline.step_backward(), None);
        timeline.speed_down();
        assert_eq!(timeline.speed(), 2.0);
    }

    #[test]
    fn timeline_caps_replay_work_per_tick() {
        let epochs = (0..64).map(|index| index as f64).collect();
        let mut timeline =
            ReplayTimeline::new(epochs, crate::tui::MAX_SPEED, false).expect("timeline");

        let first = timeline.advance_wall_time(Duration::from_secs(100));
        assert_eq!(first.len(), MAX_ADVANCE_FRAMES_PER_TICK);
        assert_eq!(first[0], 0);
        assert_eq!(first[MAX_ADVANCE_FRAMES_PER_TICK - 1], 7);

        let second = timeline.advance_wall_time(Duration::ZERO);
        assert_eq!(second.len(), MAX_ADVANCE_FRAMES_PER_TICK);
        assert_eq!(second[0], 8);
    }

    #[test]
    fn state_updates_from_fixture_replay_without_terminal() {
        let obs = read_fixture(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
        let nav = read_fixture(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
        let mut driver = ReplayDriver::from_files(&obs, &nav, 10.0, true).expect("driver");
        let mut state = TuiState::new(
            &obs.display().to_string(),
            &nav.display().to_string(),
            driver.len(),
            driver.speed(),
            driver.is_paused(),
        );
        let frame = driver
            .step_forward()
            .expect("step result")
            .expect("first frame");
        state.apply_frame(&frame);

        assert_eq!(state.current_replay_epoch, Some(1));
        assert_eq!(state.status, "solved");
        assert!(state.lat_deg.expect("lat") > 50.0);
        assert!(state.bounds.cep_m.expect("CEP") > 0.0);
        assert!(state
            .satellites
            .iter()
            .any(|sat| sat.id == "G05" && sat.used));
        assert!(state
            .satellites
            .iter()
            .any(|sat| sat.elevation_deg.is_some() && sat.azimuth_deg.is_some()));
    }

    #[test]
    fn rendering_smoke_test_has_panels_and_formatted_values() {
        let mut state = TuiState::new("site.obs", "brdc.rnx", 2, 10.0, true);
        state.current_replay_epoch = Some(1);
        state.current_raw_epoch = Some(0);
        state.observation_count = 7;
        state.epoch_time = "2020-06-25T00:00:00.000".to_string();
        state.connection_status = "streaming".to_string();
        state.status = "solved".to_string();
        state.lat_deg = Some(55.493575);
        state.lon_deg = Some(8.456829);
        state.height_m = Some(59.733);
        state.bounds = ErrorBounds {
            cep_m: Some(0.987),
            r95_m: Some(2.117),
            vertical_95_m: Some(3.477),
            sigma_e_m: Some(0.709),
            sigma_n_m: Some(0.977),
            sigma_u_m: Some(1.774),
        };
        state.satellites = vec![SatelliteSnapshot {
            id: "G05".to_string(),
            elevation_deg: Some(45.125),
            azimuth_deg: Some(123.456),
            used: true,
        }];
        state.latest_horizontal_m = Some(0.123);
        state.convergence_m.push_back(0.0);
        state.convergence_m.push_back(0.123);

        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal.draw(|frame| render(frame, &state)).expect("draw");
        let text = buffer_text(terminal.backend());

        assert!(text.contains("Solution"));
        assert!(text.contains("Satellites"));
        assert!(text.contains("Convergence"));
        assert!(text.contains("CEP"));
        assert!(text.contains("55.494 deg"));
        assert!(text.contains("0.987 m"));
        assert!(text.contains("G05"));
        assert!(text.contains("yes"));
        assert!(text.contains("streaming"));
    }

    #[test]
    fn live_driver_reads_recorded_rtcm_into_frames() {
        let nav = read_fixture(&["nav", "KMS300DNK_R_20221591000_01H_MN.rnx"]);
        let bytes = include_bytes!(
            "../../../crates/sidereon-core/tests/fixtures/rtcm/gmsd7_20121014.rtcm3"
        );
        let mut reader = Cursor::new(&bytes[..]);
        let mut driver = LiveDriver::from_tcp(
            &nav,
            1.0,
            false,
            TcpConfigInput {
                host: "offline".to_string(),
                port: 0,
            },
        )
        .expect("driver");
        let mut seen = 0usize;
        let mut frames = Vec::new();
        while frames.len() < 2 {
            let next = driver.poll_from_reader(&mut reader).expect("poll");
            if next.is_empty() {
                break;
            }
            seen += 1;
            frames.extend(next);
        }
        assert!(
            !frames.is_empty(),
            "expected at least one live frame from fixture"
        );
        assert!(seen > 0);
    }

    struct ScriptedRead {
        reads: Vec<std::io::Result<usize>>,
        index: usize,
    }

    impl ScriptedRead {
        fn with(reads: Vec<std::io::Result<usize>>) -> Self {
            Self { reads, index: 0 }
        }
    }

    impl Read for ScriptedRead {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.index >= self.reads.len() {
                return Ok(0);
            }
            let result = std::mem::replace(&mut self.reads[self.index], Ok(0));
            self.index += 1;
            match result {
                Ok(size) => {
                    for byte in buf.iter_mut().take(size) {
                        *byte = 0;
                    }
                    Ok(size)
                }
                Err(error) => Err(error),
            }
        }
    }

    #[test]
    fn live_reconnect_status_is_entered_after_stream_failure() {
        let nav = read_fixture(&["nav", "KMS300DNK_R_20221591000_01H_MN.rnx"]);
        let mut driver = LiveDriver::from_tcp(
            &nav,
            1.0,
            false,
            TcpConfigInput {
                host: "offline".to_string(),
                port: 0,
            },
        )
        .expect("driver");
        let mut stream = ScriptedRead::with(vec![Err(Error::new(
            ErrorKind::ConnectionReset,
            "network down",
        ))]);
        let frames = driver.poll_from_reader(&mut stream).expect("poll");
        assert!(frames.is_empty());
        assert!(driver.status_text().contains("reconnecting"));
    }

    fn buffer_text(backend: &TestBackend) -> String {
        let buffer = backend.buffer();
        let area = buffer.area;
        let mut out = String::new();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }
}
