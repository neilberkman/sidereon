use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sidereon::astro::passes::{GroundStation, PassPredictionOptions, UtcInstant};
use sidereon::astro::sgp4;
use sidereon::{
    horizontal_radius_at, load_rinex_nav, load_rinex_obs, load_sp3,
    metrics_from_position_covariance, parse_antex, parse_rinex_nav, parse_rinex_obs, passes,
    solve_spp, spherical_radius_at, spp_inputs_from_rinex_obs, vertical_radius_at,
};
use sidereon_core::{
    dop::PositionCovariance, smooth_track_rts, TrackCoordinateFrame, TrackFilter,
    TrackRtsHistoryBuilder,
};

use crate::qc_log_report;
use crate::solve_rinex_report;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Profile {
    Gnss,
    Astro,
    All,
}

impl std::str::FromStr for Profile {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "gnss" => Ok(Self::Gnss),
            "astro" => Ok(Self::Astro),
            "all" => Ok(Self::All),
            other => bail!("unsupported serve-mcp profile: {other}"),
        }
    }
}

impl Profile {
    fn allows_node(&self, node: &str) -> bool {
        match self {
            Self::All => true,
            Self::Gnss => {
                matches!(
                    node,
                    "ObservationFile"
                        | "BroadcastEphemeris"
                        | "Sp3"
                        | "Antex"
                        | "SolveInputs"
                        | "ReceiverSolution"
                        | "PositionErrorMetrics"
                        | "TrackFilter"
                        | "TrackRtsHistory"
                        | "TrackPoint"
                        | "Path"
                        | "FusionState"
                        | "RtkBaseline"
                        | "StaticReferenceSolution"
                        | "SsrCorrectionStore"
                        | "PassPredictionOptions"
                )
            }
            Self::Astro => {
                matches!(
                    node,
                    "Path"
                        | "TleSet"
                        | "PredictedPass"
                        | "GroundStation"
                        | "SsrCorrectionStore"
                        | "BroadcastEphemeris"
                )
            }
        }
    }

    fn allows_tool(&self, function_path: &str) -> bool {
        match self {
            Self::All => true,
            Self::Gnss => {
                matches!(
                    function_path,
                    "solve_rinex" | "qc_log" | "error_metrics" | "inspect_file" | "clean_track"
                )
            }
            Self::Astro => matches!(function_path, "predict_passes"),
        }
    }
}

#[derive(Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    result: Option<Value>,
    error: Option<Value>,
}

#[derive(Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Deserialize)]
struct RpcRequest {
    method: String,
    id: Option<Value>,
    params: Option<Value>,
}

#[derive(Clone)]
struct ToolInvocation {
    name: &'static str,
    function_path: &'static str,
    description: &'static str,
    schema: Value,
    function: fn(Value) -> Result<Value>,
}

#[derive(Clone)]
struct CapabilityEdge {
    from: &'static str,
    to: &'static str,
    function_path: &'static str,
    name: &'static str,
    description: &'static str,
    invokable: bool,
}

struct CapabilityGraph {
    profile: Profile,
    nodes: Vec<&'static str>,
    edges: Vec<CapabilityEdge>,
    tool_invocations: HashMap<&'static str, ToolInvocation>,
}

pub fn serve_mcp_command(profile: &str) -> Result<()> {
    let profile = profile.parse::<Profile>()?;
    let graph = CapabilityGraph::v1(profile);

    let stdin = io::stdin();
    let mut out = io::stdout();

    for line in BufReader::new(stdin.lock()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: RpcRequest = serde_json::from_str(&line).context("parse MCP json")?;
        // JSON-RPC notifications (no id) never receive responses; MCP clients
        // send notifications/initialized right after the handshake.
        if request.id.is_none() {
            continue;
        }
        let response = handle_request(request, &graph);
        let encoded = serde_json::to_string(&response)?;
        out.write_all(encoded.as_bytes())?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    Ok(())
}

fn handle_request(request: RpcRequest, graph: &CapabilityGraph) -> RpcResponse {
    let params = request.params.unwrap_or_else(|| json!({}));
    match request.method.as_str() {
        "initialize" => {
            let requested = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or("2024-11-05");
            RpcResponse {
                jsonrpc: "2.0",
                id: request.id,
                result: Some(json!({
                    "protocolVersion": requested,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {
                        "name": "sidereon",
                        "version": env!("CARGO_PKG_VERSION"),
                        "profile": format!("{:?}", graph.profile),
                    },
                })),
                error: None,
            }
        }
        "ping" => RpcResponse {
            jsonrpc: "2.0",
            id: request.id,
            result: Some(json!("pong")),
            error: None,
        },
        "tools/list" => RpcResponse {
            jsonrpc: "2.0",
            id: request.id,
            result: Some(json!({"tools": graph.tool_list()})),
            error: None,
        },
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match graph
                .tool_invocations
                .get(name)
                .filter(|tool| graph.profile.allows_tool(tool.function_path))
            {
                Some(tool) => match (tool.function)(arguments) {
                    Ok(result) => RpcResponse {
                        jsonrpc: "2.0",
                        id: request.id,
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&result)
                                    .unwrap_or_else(|_| result.to_string()),
                            }],
                            "structuredContent": result,
                            "isError": false,
                        })),
                        error: None,
                    },
                    Err(error) => RpcResponse {
                        jsonrpc: "2.0",
                        id: request.id,
                        result: Some(json!({
                            "content": [{"type": "text", "text": error.to_string()}],
                            "isError": true,
                        })),
                        error: None,
                    },
                },
                None => RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: None,
                    error: Some(json!(RpcError::new(-32601, "tool not found"))),
                },
            }
        }
        "resources/list" => RpcResponse {
            jsonrpc: "2.0",
            id: request.id,
            result: Some(json!({"resources": Layer3Resources::all_list()})),
            error: None,
        },
        "resources/read" => {
            let uri = params.get("uri").and_then(Value::as_str).unwrap_or("");
            if let Some(value) = Layer3Resources::read(uri, graph) {
                RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: Some(value),
                    error: None,
                }
            } else {
                RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: None,
                    error: Some(json!(RpcError::new(-32001, "unknown resource"))),
                }
            }
        }
        "capability/map" => RpcResponse {
            jsonrpc: "2.0",
            id: request.id,
            result: Some(graph.capability_map()),
            error: None,
        },
        "capability/operations_on" => {
            let node = params.get("node").and_then(Value::as_str).unwrap_or("");
            RpcResponse {
                jsonrpc: "2.0",
                id: request.id,
                result: Some(json!({"operations": graph.operations_on(node)})),
                error: None,
            }
        }
        "capability/from_format" => {
            let description = params
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            RpcResponse {
                jsonrpc: "2.0",
                id: request.id,
                result: Some(json!({"paths": graph.from_format(description)})),
                error: None,
            }
        }
        "capability/path" => {
            let from = params.get("from").and_then(Value::as_str).unwrap_or("");
            let to = params.get("to").and_then(Value::as_str).unwrap_or("");
            RpcResponse {
                jsonrpc: "2.0",
                id: request.id,
                result: Some(json!({"path": graph.path(from, to)})),
                error: None,
            }
        }
        "capability/describe" => {
            let function_path = params
                .get("function_path")
                .and_then(Value::as_str)
                .unwrap_or("");
            if let Some(schema) = graph.describe(function_path) {
                RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: Some(schema),
                    error: None,
                }
            } else {
                RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: None,
                    error: Some(json!(RpcError::new(-32601, "unknown function path"))),
                }
            }
        }
        "capability/invoke" => {
            let function_path = params
                .get("function_path")
                .and_then(Value::as_str)
                .unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match graph.invoke(function_path, arguments) {
                Ok(result) => RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: Some(json!({"status": "ok", "result": result})),
                    error: None,
                },
                Err(error) => RpcResponse {
                    jsonrpc: "2.0",
                    id: request.id,
                    result: Some(json!({
                        "function_path": function_path,
                        "status": "not invokable in v1",
                        "graph_entry": graph.entry_for_path(function_path),
                        "error": error.to_string(),
                    })),
                    error: Some(json!(RpcError::new(-32000, "not invokable in v1"))),
                },
            }
        }
        _ => RpcResponse {
            jsonrpc: "2.0",
            id: request.id,
            result: None,
            error: Some(json!(RpcError::new(-32601, "method not found"))),
        },
    }
}

impl CapabilityGraph {
    fn v1(profile: Profile) -> Self {
        let _load_rinex_obs = |path: &str| load_rinex_obs(path);
        let _load_rinex_nav = |path: &str| load_rinex_nav(path);
        let _load_sp3: fn(&[u8]) -> sidereon::Result<sidereon::ephemeris::Sp3> = load_sp3;
        let _ = parse_rinex_obs;
        let _ = parse_rinex_nav;
        let _ = parse_antex;
        let _ = sgp4::parse_tle_file;
        let _ = spp_inputs_from_rinex_obs::<dyn sidereon::positioning::RinexSppAssemblySource>;
        let _ = solve_spp;
        let _ = metrics_from_position_covariance;
        let _ = TrackFilter::from_position3;
        let _ = TrackFilter::predict_recorded;
        let _ = TrackFilter::update_position_recorded;
        let _ = smooth_track_rts;
        let _ = passes::predict_passes;

        let nodes = vec![
            "Path",
            "ObservationFile",
            "BroadcastEphemeris",
            "Sp3",
            "Antex",
            "TleSet",
            "SolveInputs",
            "ReceiverSolution",
            "PositionErrorMetrics",
            "TrackPoint",
            "TrackFilter",
            "TrackRtsHistory",
            "FusionState",
            "RtkBaseline",
            "StaticReferenceSolution",
            "SsrCorrectionStore",
            "GroundStation",
            "PredictedPass",
            "PassPredictionOptions",
        ];

        let edges = vec![
            CapabilityEdge {
                from: "Path",
                to: "ObservationFile",
                function_path: "sidereon::load_rinex_obs",
                name: "load_rinex_obs",
                description: "Load RINEX OBS from path",
                invokable: false,
            },
            CapabilityEdge {
                from: "Path",
                to: "BroadcastEphemeris",
                function_path: "sidereon::load_rinex_nav",
                name: "load_rinex_nav",
                description: "Load RINEX NAV from path",
                invokable: false,
            },
            CapabilityEdge {
                from: "Path",
                to: "Sp3",
                function_path: "sidereon::load_sp3",
                name: "load_sp3",
                description: "Load SP3 precise orbits from bytes",
                invokable: false,
            },
            CapabilityEdge {
                from: "Path",
                to: "Antex",
                function_path: "sidereon::parse_antex",
                name: "parse_antex",
                description: "Parse ANTEX from text",
                invokable: false,
            },
            CapabilityEdge {
                from: "Path",
                to: "TleSet",
                function_path: "sidereon::astro::sgp4::parse_tle_file",
                name: "parse_tle_file",
                description: "Parse TLE file with tolerant name-line handling",
                invokable: false,
            },
            CapabilityEdge {
                from: "ObservationFile",
                to: "SolveInputs",
                function_path: "sidereon::spp_inputs_from_rinex_obs",
                name: "spp_inputs_from_rinex_obs",
                description: "Build SPP epoch inputs from RINEX OBS",
                invokable: false,
            },
            CapabilityEdge {
                from: "BroadcastEphemeris",
                to: "SolveInputs",
                function_path: "sidereon::spp_inputs_from_rinex_obs",
                name: "spp_inputs_from_rinex_obs",
                description: "Build SPP inputs using broadcast context",
                invokable: false,
            },
            CapabilityEdge {
                from: "Sp3",
                to: "SolveInputs",
                function_path: "sidereon::positioning::RinexSppSource::with_broadcast_context",
                name: "with_broadcast_context",
                description: "Build SP3 context for SPP assembly",
                invokable: false,
            },
            CapabilityEdge {
                from: "SolveInputs",
                to: "ReceiverSolution",
                function_path: "sidereon::solve_spp",
                name: "solve_spp",
                description: "Solve single SPP epoch",
                invokable: false,
            },
            CapabilityEdge {
                from: "ReceiverSolution",
                to: "PositionErrorMetrics",
                function_path: "sidereon::metrics_from_position_covariance",
                name: "metrics_from_position_covariance",
                description: "Compute CEP/R95 bounds from covariance",
                invokable: false,
            },
            CapabilityEdge {
                from: "TrackPoint",
                to: "TrackFilter",
                function_path: "sidereon_core::estimation::track::TrackFilter::from_position3",
                name: "from_position3",
                description: "Start position-only filter",
                invokable: false,
            },
            CapabilityEdge {
                from: "TrackFilter",
                to: "TrackRtsHistory",
                function_path: "sidereon_core::estimation::track::TrackFilter::predict_recorded",
                name: "predict_recorded",
                description: "Predict interval and append to RTS history",
                invokable: false,
            },
            CapabilityEdge {
                from: "TrackRtsHistory",
                to: "TrackRtsHistory",
                function_path:
                    "sidereon_core::estimation::track::TrackFilter::update_position_recorded",
                name: "update_position_recorded",
                description: "Apply measured position and append to RTS history",
                invokable: false,
            },
            CapabilityEdge {
                from: "TrackRtsHistory",
                to: "TrackRtsHistory",
                function_path: "sidereon_core::estimation::track::smooth_track_rts",
                name: "smooth_track_rts",
                description: "RTS smooth complete track history",
                invokable: false,
            },
            CapabilityEdge {
                from: "TleSet",
                to: "PredictedPass",
                function_path: "sidereon::passes::predict_passes",
                name: "predict_passes",
                description: "Predict station pass windows from elements",
                invokable: false,
            },
            CapabilityEdge {
                from: "GroundStation",
                to: "PredictedPass",
                function_path: "sidereon::passes::predict_passes",
                name: "predict_passes",
                description: "Predict station pass windows from station context",
                invokable: false,
            },
        ];

        let invocations = [
            (
                "solve_rinex",
                ToolInvocation {
                    name: "solve_rinex",
                    function_path: "solve_rinex",
                    description: "Run SPP over RINEX OBS with NAV and optional SP3",
                    schema: solve_rinex_schema(),
                    function: solve_rinex_invocation,
                },
            ),
            (
                "qc_log",
                ToolInvocation {
                    name: "qc_log",
                    function_path: "qc_log",
                    description: "RINEX OBS lint and QC",
                    schema: qc_log_schema(),
                    function: qc_log_invocation,
                },
            ),
            (
                "error_metrics",
                ToolInvocation {
                    name: "error_metrics",
                    function_path: "error_metrics",
                    description: "Covariance-derived position metrics for ENU covariance",
                    schema: error_metrics_schema(),
                    function: error_metrics_invocation,
                },
            ),
            (
                "inspect_file",
                ToolInvocation {
                    name: "inspect_file",
                    function_path: "inspect_file",
                    description: "Inspect OBS/NAV/SP3/ANTEX/TLE from file",
                    schema: inspect_file_schema(),
                    function: inspect_file_invocation,
                },
            ),
            (
                "clean_track",
                ToolInvocation {
                    name: "clean_track",
                    function_path: "clean_track",
                    description: "Filter and RTS-smooth a 3D point list",
                    schema: clean_track_schema(),
                    function: clean_track_invocation,
                },
            ),
            (
                "predict_passes",
                ToolInvocation {
                    name: "predict_passes",
                    function_path: "predict_passes",
                    description: "Predict visible passes from TLE and station location",
                    schema: predict_passes_schema(),
                    function: predict_passes_invocation,
                },
            ),
        ]
        .into_iter()
        .collect();

        Self {
            profile,
            nodes,
            edges,
            tool_invocations: invocations,
        }
    }

    fn tool_list(&self) -> Vec<Value> {
        self.tool_invocations
            .values()
            .filter(|tool| self.profile.allows_tool(tool.function_path))
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema,
                })
            })
            .collect()
    }

    fn capability_map(&self) -> Value {
        let nodes = self
            .nodes
            .iter()
            .filter(|node| self.profile.allows_node(node))
            .map(|node| json!({"node": node}));

        let edges = self
            .edges
            .iter()
            .filter(|edge| self.profile.allows_node(edge.from) && self.profile.allows_node(edge.to))
            .map(|edge| {
                json!({
                    "from": edge.from,
                    "to": edge.to,
                    "function_path": edge.function_path,
                    "name": edge.name,
                    "description": edge.description,
                    "invokable": edge.invokable,
                })
            });

        json!({"nodes": nodes.collect::<Vec<_>>(), "edges": edges.collect::<Vec<_>>()})
    }

    fn operations_on(&self, node: &str) -> Vec<Value> {
        self.edges
            .iter()
            .filter(|edge| {
                (edge.from == node || edge.to == node)
                    && self.profile.allows_node(edge.from)
                    && self.profile.allows_node(edge.to)
            })
            .map(|edge| {
                json!({
                    "from": edge.from,
                    "to": edge.to,
                    "function_path": edge.function_path,
                    "name": edge.name,
                    "description": edge.description,
                    "invokable": edge.invokable,
                })
            })
            .collect()
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_format(&self, needle: &str) -> Vec<String> {
        let needle = needle.to_lowercase();
        self.edges
            .iter()
            .filter(|edge| {
                self.profile.allows_node(edge.from)
                    && self.profile.allows_node(edge.to)
                    && (edge.from.to_lowercase().contains(&needle)
                        || edge.to.to_lowercase().contains(&needle)
                        || edge.name.to_lowercase().contains(&needle)
                        || edge.description.to_lowercase().contains(&needle))
            })
            .map(|edge| edge.function_path.to_string())
            .collect()
    }

    fn path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.profile.allows_node(from) || !self.profile.allows_node(to) {
            return None;
        }

        let mut queue = VecDeque::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut via: HashMap<String, String> = HashMap::new();

        queue.push_back(from.to_string());
        prev.insert(from.to_string(), String::new());

        while let Some(current) = queue.pop_front() {
            if current == to {
                let mut out = Vec::new();
                let mut cursor = to.to_string();
                while let Some(parent) = prev.get(&cursor) {
                    if parent.is_empty() {
                        break;
                    }
                    out.push(via.get(&cursor).cloned().unwrap_or_default());
                    cursor = parent.clone();
                }
                out.reverse();
                return Some(out);
            }

            for edge in self
                .edges
                .iter()
                .filter(|edge| edge.from == current && self.profile.allows_node(edge.to))
            {
                if !prev.contains_key(edge.to) {
                    prev.insert(edge.to.to_string(), current.clone());
                    via.insert(edge.to.to_string(), edge.function_path.to_string());
                    queue.push_back(edge.to.to_string());
                }
            }
        }

        None
    }

    fn describe(&self, function_path: &str) -> Option<Value> {
        self.tool_invocations
            .values()
            .find(|tool| tool.function_path == function_path)
            .map(|tool| json!({"tool": tool.name, "schema": tool.schema, "function_path": tool.function_path}))
            .or_else(|| {
                self.edges
                    .iter()
                    .find(|edge| edge.function_path == function_path)
                    .map(|edge| json!({
                        "function_path": edge.function_path,
                        "name": edge.name,
                        "from": edge.from,
                        "to": edge.to,
                        "description": edge.description
                    }))
            })
    }

    fn entry_for_path(&self, function_path: &str) -> Value {
        self.describe(function_path)
            .unwrap_or_else(|| json!({"function_path": function_path, "status": "unregistered"}))
    }

    fn invoke(&self, function_path: &str, params: Value) -> Result<Value> {
        let tool = self
            .tool_invocations
            .values()
            .find(|tool| tool.function_path == function_path)
            .ok_or_else(|| anyhow::anyhow!("{} not invokable in v1", function_path))?;

        if !self.profile.allows_tool(tool.function_path) {
            bail!("{} not invokable in v1", function_path)
        }

        (tool.function)(params)
    }
}

#[derive(Debug, Deserialize)]
struct SolveRinexParams {
    obs_path: String,
    nav_path: Option<String>,
    sp3_path: Option<String>,
}

fn solve_rinex_invocation(raw: Value) -> Result<Value> {
    let params: SolveRinexParams = serde_json::from_value(raw)?;
    let nav_path = params
        .nav_path
        .ok_or_else(|| anyhow::anyhow!("nav_path is required for solve_rinex"))?;
    let report = solve_rinex_report(
        Path::new(&params.obs_path),
        Path::new(&nav_path),
        params.sp3_path.as_deref().map(Path::new),
    )?;
    Ok(serde_json::to_value(report)?)
}

fn solve_rinex_schema() -> Value {
    json!({
        "type": "object",
        "required": ["obs_path", "nav_path"],
        "properties": {
            "obs_path": {"type": "string"},
            "nav_path": {"type": "string"},
            "sp3_path": {"type": ["string", "null"]}
        }
    })
}

#[derive(Debug, Deserialize)]
struct QcLogParams {
    obs_path: String,
}

fn qc_log_invocation(raw: Value) -> Result<Value> {
    let params: QcLogParams = serde_json::from_value(raw)?;
    let report = qc_log_report(Path::new(&params.obs_path))?;
    Ok(serde_json::to_value(report)?)
}

fn qc_log_schema() -> Value {
    json!({
        "type": "object",
        "required": ["obs_path"],
        "properties": {
            "obs_path": {"type": "string"}
        }
    })
}

#[derive(Debug, Deserialize)]
struct ErrorMetricsParams {
    enu_covariance_3x3: [[f64; 3]; 3],
}

fn error_metrics_invocation(raw: Value) -> Result<Value> {
    let params: ErrorMetricsParams = serde_json::from_value(raw)?;
    let covariance = PositionCovariance {
        ecef_m2: params.enu_covariance_3x3,
        enu_m2: params.enu_covariance_3x3,
    };

    match metrics_from_position_covariance(&covariance) {
        Ok(metrics) => {
            let horizontal = match horizontal_radius_at(params.enu_covariance_3x3, 0.95) {
                Ok(value) => value,
                Err(err) => {
                    return Ok(json!({
                        "validity_flag": false,
                        "input": {
                            "enu_covariance_3x3": params.enu_covariance_3x3,
                            "probability": 0.95,
                        },
                        "error": format!("{:?}", err),
                    }));
                }
            };
            let vertical = match vertical_radius_at(params.enu_covariance_3x3[2][2], 0.95) {
                Ok(value) => value,
                Err(err) => {
                    return Ok(json!({
                        "validity_flag": false,
                        "input": {
                            "enu_covariance_3x3": params.enu_covariance_3x3,
                            "probability": 0.95,
                        },
                        "error": format!("{:?}", err),
                    }));
                }
            };
            let spherical = match spherical_radius_at(params.enu_covariance_3x3, 0.95) {
                Ok(value) => value,
                Err(err) => {
                    return Ok(json!({
                        "validity_flag": false,
                        "input": {
                            "enu_covariance_3x3": params.enu_covariance_3x3,
                            "probability": 0.95,
                        },
                        "error": format!("{:?}", err),
                    }));
                }
            };
            Ok(json!({
                "validity_flag": true,
                "input": {
                    "enu_covariance_3x3": params.enu_covariance_3x3,
                    "probability": 0.95,
                },
                "metrics": {
                    "cep_m": metrics.cep_m.radius_m,
                    "r95_m": metrics.r95_m.radius_m,
                    "r99_m": metrics.r99_m.radius_m,
                    "sigma_e_m": metrics.sigma_e_m,
                    "sigma_n_m": metrics.sigma_n_m,
                    "sigma_u_m": metrics.sigma_u_m,
                    "ellipse_semi_major_m": metrics.ellipse.semi_major_m,
                    "ellipse_semi_minor_m": metrics.ellipse.semi_minor_m,
                    "ellipse_orientation_deg": metrics.ellipse.orientation_rad.to_degrees(),
                    "drms_m": metrics.drms_m,
                    "two_drms_m": metrics.two_drms_m,
                    "vep_m": metrics.vep_m,
                    "sep_m": metrics.sep_m.radius_m,
                    "mrse_m": metrics.mrse_m,
                    "horizontal_radius_m": horizontal.radius_m,
                    "vertical_radius_m": vertical,
                    "spherical_radius_m": spherical.radius_m,
                }
            }))
        }
        Err(error) => Ok(json!({
            "validity_flag": false,
            "input": {
                "enu_covariance_3x3": params.enu_covariance_3x3,
                "probability": 0.95,
            },
            "error": format!("{:?}", error),
        })),
    }
}

fn error_metrics_schema() -> Value {
    json!({
        "type": "object",
        "required": ["enu_covariance_3x3"],
        "properties": {
            "enu_covariance_3x3": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"}
                }
            }
        }
    })
}

#[derive(Debug, Deserialize)]
struct InspectFileParams {
    path: String,
}

fn inspect_file_invocation(raw: Value) -> Result<Value> {
    let params: InspectFileParams = serde_json::from_value(raw)?;
    let path = Path::new(&params.path);
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).ok();

    if let Some(text) = text {
        if let Ok(_obs) = parse_rinex_obs(text) {
            return Ok(json!({"path": path.display().to_string(), "type": "RINEX OBS"}));
        }
        if let Ok(_nav) = parse_rinex_nav(text) {
            return Ok(json!({"path": path.display().to_string(), "type": "RINEX NAV"}));
        }
        if let Ok(antex) = parse_antex(text) {
            if !antex.antennas.is_empty() {
                return Ok(json!({
                    "path": path.display().to_string(),
                    "type": "ANTEX",
                    "antennas": antex.antennas.len()
                }));
            }
        }
        if let Some(info) = inspect_tle_text(text) {
            return Ok(json!({"path": path.display().to_string(), "type": "TLE", "info": info}));
        }
    }

    if let Ok(sp3) = load_sp3(&bytes) {
        return Ok(json!({
            "path": path.display().to_string(),
            "type": "SP3",
            "epoch_count": sp3.epoch_count(),
            "satellite_count": sp3.satellites().len(),
        }));
    }

    bail!("unrecognized file type: {}", path.display())
}

fn inspect_file_schema() -> Value {
    json!({
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string"}
        }
    })
}

fn inspect_tle_text(text: &str) -> Option<Value> {
    let mut satellites = Vec::new();
    let mut skipped = 0usize;

    for pair in tle_pairs_from_text(text) {
        match sidereon::tle::parse(pair.0, pair.1) {
            Ok(parsed) => {
                let catalog = parsed.elements.catalog_number;
                satellites.push(json!({"catalog": catalog}));
            }
            Err(_) => skipped += 1,
        }
    }

    if satellites.is_empty() && skipped == 0 {
        None
    } else {
        Some(json!({"pairs": satellites.len(), "skipped": skipped, "entries": satellites}))
    }
}

#[derive(Debug, Deserialize)]
struct CleanTrackPoint {
    t_s: f64,
    position_m: [f64; 3],
    covariance_m2: [[f64; 3]; 3],
}

#[derive(Debug, Deserialize)]
struct CleanTrackParams {
    points: Vec<CleanTrackPoint>,
    initial_velocity_m2_s: Option<f64>,
    acceleration_variance_m2_s3: Option<f64>,
}

fn clean_track_invocation(raw: Value) -> Result<Value> {
    let params: CleanTrackParams = serde_json::from_value(raw)?;
    let first = params
        .points
        .first()
        .ok_or_else(|| anyhow::anyhow!("clean_track requires at least one point"))?;

    let mut filter = TrackFilter::from_position3(
        TrackCoordinateFrame::Ecef,
        first.t_s,
        first.position_m,
        first.covariance_m2,
        params.initial_velocity_m2_s.unwrap_or(4.0),
        params.acceleration_variance_m2_s3.unwrap_or(1.0e-4),
    )
    .map_err(|error| anyhow::anyhow!("build track filter: {error}"))?;

    let mut history = TrackRtsHistoryBuilder::from_filter(&filter)
        .map_err(|error| anyhow::anyhow!("start history: {error}"))?;

    let mut filtered = Vec::new();
    filtered.push(json!({"t_s": first.t_s, "position_m": first.position_m}));

    for point in &params.points[1..] {
        let dt = point.t_s - filter.state().t_s;
        if dt <= 0.0 {
            bail!("points must be strictly increasing in t_s");
        }
        filter
            .predict_recorded(dt, &mut history)
            .map_err(|error| anyhow::anyhow!("predict: {error}"))?;
        filter
            .update_position_recorded(
                &point.position_m,
                &matrix3_rows(point.covariance_m2),
                &mut history,
            )
            .map_err(|error| anyhow::anyhow!("update: {error}"))?;
        filtered.push(json!({"t_s": point.t_s, "position_m": filter.state().position3_m().unwrap_or([0.0; 3])}));
    }

    let smoothed = smooth_track_rts(&history.finish()?)
        .map_err(|error| anyhow::anyhow!("smooth track: {error}"))?;
    let smoothed_points = smoothed
        .epochs
        .iter()
        .map(|epoch| {
            json!({
                "t_s": epoch.t_s,
                "position_m": epoch.state.position3_m().unwrap_or([0.0; 3]),
                "position_covariance_m2": epoch.state.position_covariance3_m2().ok(),
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "filtered": filtered,
        "smoothed": smoothed_points,
        "filtered_count": filtered.len(),
        "smoothed_count": smoothed_points.len(),
    }))
}

fn clean_track_schema() -> Value {
    json!({
        "type": "object",
        "required": ["points"],
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["t_s", "position_m", "covariance_m2"],
                    "properties": {
                        "t_s": {"type": "number"},
                        "position_m": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"}
                        },
                        "covariance_m2": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "number"}
                            }
                        }
                    }
                }
            },
            "initial_velocity_m2_s": {"type": "number", "minimum": 0.0},
            "acceleration_variance_m2_s3": {"type": "number", "minimum": 0.0}
        }
    })
}

#[derive(Debug, Deserialize)]
struct PredictPassesParams {
    tle_path: String,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    hours: f64,
}

fn predict_passes_invocation(raw: Value) -> Result<Value> {
    let params: PredictPassesParams = serde_json::from_value(raw)?;
    let path = Path::new(&params.tle_path);
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    if params.hours.partial_cmp(&0.0) != Some(Ordering::Greater) {
        bail!("hours must be > 0");
    }

    let now = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system time")?
            .as_micros(),
    )
    .context("unix micros out of range")?;
    let start = UtcInstant::from_unix_microseconds(now);
    let end =
        UtcInstant::from_unix_microseconds(now + (params.hours * 3_600_000_000.0).round() as i64);

    let station = GroundStation {
        latitude_deg: params.lat_deg,
        longitude_deg: params.lon_deg,
        altitude_m: params.height_m,
    };
    let mut options = PassPredictionOptions::default();
    options.min_elevation_deg = 0.0;
    options.step_seconds = 20;

    let mut satellites = Vec::new();
    for (index, (line1, line2)) in tle_pairs_from_text(&text).into_iter().enumerate() {
        let parsed = sidereon::tle::parse(line1, line2).context("parse tle")?;
        let elements = parsed
            .elements
            .to_element_set()
            .context("tle to element set")?;
        let passes = passes::predict_passes(&elements, station, start, end, options)
            .context("predict passes")?;
        satellites.push(json!({
            "index": index,
            "catalog": parsed.elements.catalog_number.to_string(),
            "pass_count": passes.len(),
            "passes": passes
                .into_iter()
                .map(|entry| json!({
                    "rise_unix_microseconds": entry.rise.unix_microseconds(),
                    "set_unix_microseconds": entry.set.unix_microseconds(),
                    "max_elevation_deg": entry.max_elevation_deg,
                    "max_elevation_time_unix_microseconds": entry.max_elevation_time.unix_microseconds()
                }))
                .collect::<Vec<_>>()
        }));
    }

    Ok(json!({
        "station": {
            "lat_deg": params.lat_deg,
            "lon_deg": params.lon_deg,
            "height_m": params.height_m,
        },
        "window_hours": params.hours,
        "window": {
            "start_unix_microseconds": start.unix_microseconds(),
            "end_unix_microseconds": end.unix_microseconds(),
        },
        "options": {
            "min_elevation_deg": options.min_elevation_deg,
            "step_seconds": options.step_seconds,
        },
        "satellites": satellites,
        "satellite_count": satellites.len(),
    }))
}

fn predict_passes_schema() -> Value {
    json!({
        "type": "object",
        "required": ["tle_path", "lat_deg", "lon_deg", "height_m", "hours"],
        "properties": {
            "tle_path": {"type": "string"},
            "lat_deg": {"type": "number", "minimum": -90.0, "maximum": 90.0},
            "lon_deg": {"type": "number", "minimum": -180.0, "maximum": 180.0},
            "height_m": {"type": "number"},
            "hours": {"type": "number", "exclusiveMinimum": 0.0}
        }
    })
}

fn matrix3_rows(matrix: [[f64; 3]; 3]) -> Vec<Vec<f64>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}

fn tle_pairs_from_text(text: &str) -> Vec<(&str, &str)> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx + 1 < lines.len() {
        if lines[idx].starts_with('1') && lines[idx + 1].starts_with('2') {
            out.push((lines[idx], lines[idx + 1]));
            idx += 2;
            continue;
        }
        if idx + 2 < lines.len()
            && lines[idx + 1].starts_with('1')
            && lines[idx + 2].starts_with('2')
        {
            out.push((lines[idx + 1], lines[idx + 2]));
            idx += 3;
            continue;
        }
        idx += 1;
    }
    out
}

#[derive(Clone)]
struct ResourceDoc {
    uri: &'static str,
    name: &'static str,
    text: &'static str,
}

struct Layer3Resources;

impl Layer3Resources {
    fn all() -> Vec<ResourceDoc> {
        vec![
            ResourceDoc {
                uri: "sidereon://docs/concepts/frames-time-scales",
                name: "Frames and time scales",
                text: "Coordinate paths use ENU and ECEF in core types. Time inputs are UTC and pass planning uses internal time scales for prediction.",
            },
            ResourceDoc {
                uri: "sidereon://docs/concepts/cep-r95",
                name: "CEP and R95",
                text: "Derived metrics include CEP, R95, R99, DRMS, 2DRMS and related metrics. validity_flag is false for invalid covariance input.",
            },
            ResourceDoc {
                uri: "sidereon://docs/concepts/product-expectations",
                name: "Product file expectations",
                text: "solve_rinex consumes RINEX OBS + NAV and optional SP3. inspect_file accepts RINEX OBS/NAV, SP3, ANTEX, and TLE formats.",
            },
            ResourceDoc {
                uri: "sidereon://docs/capability-map",
                name: "Capability map",
                text: "Call capability/map for the current profile nodes and edges.",
            },
            ResourceDoc {
                uri: "sidereon://docs/capability-map.json",
                name: "Capability map JSON",
                text: "Same data as capability/map as plain JSON.",
            },
        ]
    }

    fn all_list() -> Vec<Value> {
        Self::all()
            .into_iter()
            .map(|doc| json!({"uri": doc.uri, "name": doc.name}))
            .collect()
    }

    fn read(uri: &str, graph: &CapabilityGraph) -> Option<Value> {
        match uri {
            "sidereon://docs/capability-map" | "sidereon://docs/capability-map.json" => {
                Some(graph.capability_map())
            }
            _ => Self::all()
                .into_iter()
                .find(|doc| doc.uri == uri)
                .map(|doc| json!({"uri": doc.uri, "name": doc.name, "text": doc.text})),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn fixture(parts: &[&str]) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../sidereon-core/tests/fixtures");
        for part in parts {
            path.push(part);
        }
        path
    }

    #[test]
    fn solve_rinex_tool_matches_cli_output() {
        let obs = fixture(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
        let nav = fixture(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
        let sp3 = fixture(&["sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"]);

        let cli = solve_rinex_report(&obs, &nav, Some(&sp3)).expect("cli solve");
        let tool = solve_rinex_invocation(json!({
            "obs_path": obs.to_str().expect("obs"),
            "nav_path": nav.to_str().expect("nav"),
            "sp3_path": sp3.to_str().expect("sp3")
        }))
        .expect("tool");

        assert_eq!(serde_json::to_value(cli).expect("to json"), tool);
    }

    #[test]
    fn graph_integrity_test() {
        let graph = CapabilityGraph::v1(Profile::All);

        let nodes: HashSet<&'static str> = graph.nodes.iter().copied().collect();
        for edge in &graph.edges {
            assert!(nodes.contains(edge.from));
            assert!(nodes.contains(edge.to));
        }

        for edge in graph.edges.iter().filter(|edge| edge.invokable) {
            assert!(
                graph
                    .tool_invocations
                    .values()
                    .any(|tool| tool.function_path == edge.function_path),
                "{} missing dispatch",
                edge.function_path
            );
        }

        let path = graph
            .path("ObservationFile", "PositionErrorMetrics")
            .expect("path");
        assert_eq!(
            path,
            vec![
                "sidereon::spp_inputs_from_rinex_obs",
                "sidereon::solve_spp",
                "sidereon::metrics_from_position_covariance",
            ]
        );
    }

    #[test]
    fn schema_roundtrip() {
        let graph = CapabilityGraph::v1(Profile::All);
        let schema = graph.describe("solve_rinex").expect("schema");
        let _: Value = serde_json::from_value(schema).expect("json");
    }

    fn request(method: &str, params: Value) -> RpcRequest {
        RpcRequest {
            method: method.to_string(),
            id: Some(json!(1)),
            params: Some(params),
        }
    }

    #[test]
    fn mcp_initialize_returns_spec_shape() {
        let graph = CapabilityGraph::v1(Profile::All);
        let response = handle_request(
            request("initialize", json!({"protocolVersion": "2024-11-05"})),
            &graph,
        );
        let result = response.result.expect("result");
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
        assert_eq!(result["serverInfo"]["name"], "sidereon");
        assert!(result["serverInfo"]["version"].is_string());
    }

    #[test]
    fn mcp_tools_list_uses_camel_case_input_schema() {
        let graph = CapabilityGraph::v1(Profile::All);
        let response = handle_request(request("tools/list", json!({})), &graph);
        let result = response.result.expect("result");
        let tools = result["tools"].as_array().expect("tools array");
        assert!(!tools.is_empty());
        for tool in tools {
            assert!(
                tool["inputSchema"].is_object(),
                "missing inputSchema: {tool}"
            );
            assert!(tool.get("input_schema").is_none());
        }
        let solve = tools
            .iter()
            .find(|tool| tool["name"] == "solve_rinex")
            .expect("solve_rinex listed");
        let required = solve["inputSchema"]["required"]
            .as_array()
            .expect("required");
        assert!(required.contains(&json!("nav_path")));
    }

    #[test]
    fn mcp_tools_call_wraps_results_in_content() {
        let graph = CapabilityGraph::v1(Profile::All);
        let response = handle_request(
            request(
                "tools/call",
                json!({"name": "error_metrics", "arguments": {
                    "enu_covariance_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                }}),
            ),
            &graph,
        );
        let result = response.result.expect("result");
        assert_eq!(result["isError"], false);
        let content = result["content"].as_array().expect("content array");
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].is_string());
        assert!(result["structuredContent"].is_object());
    }

    #[test]
    fn mcp_tool_failures_are_is_error_content_not_protocol_errors() {
        let graph = CapabilityGraph::v1(Profile::All);
        let response = handle_request(
            request(
                "tools/call",
                json!({"name": "solve_rinex", "arguments": {
                    "obs_path": "/nonexistent.obs", "nav_path": "/nonexistent.nav",
                }}),
            ),
            &graph,
        );
        assert!(
            response.error.is_none(),
            "tool failure must not be a protocol error"
        );
        let result = response.result.expect("result");
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"].is_string());
    }

    #[test]
    fn mcp_unknown_tool_is_a_protocol_error() {
        let graph = CapabilityGraph::v1(Profile::All);
        let response = handle_request(
            request(
                "tools/call",
                json!({"name": "no_such_tool", "arguments": {}}),
            ),
            &graph,
        );
        assert!(response.result.is_none());
        assert!(response.error.is_some());
    }
}
