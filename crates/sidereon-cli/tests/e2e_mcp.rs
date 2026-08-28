use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Generous because the server is a debug-profile child process and the suite
/// runs many tests in parallel; a tight bound turns scheduling delay into a
/// spurious failure. It still bounds a genuinely hung server.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

#[test]
fn serve_mcp_stdio_conforms_to_json_rpc_flow() -> TestResult {
    let mut server = McpServer::spawn()?;

    server.send(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "sidereon-e2e", "version": "0"}
        }
    }))?;
    let initialize = server.recv(RESPONSE_TIMEOUT)?;
    assert_eq!(initialize["jsonrpc"], "2.0");
    assert_eq!(initialize["id"], 1);
    let result = &initialize["result"];
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"]["tools"].is_object());
    assert!(result["capabilities"]["resources"].is_object());
    assert_eq!(result["serverInfo"]["name"], "sidereon");
    assert!(result["serverInfo"]["version"].is_string());

    server.send(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }))?;
    assert!(
        server.recv_optional(Duration::from_millis(200))?.is_none(),
        "notifications/initialized must not receive a response"
    );

    server.send(json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }))?;
    let tools_response = server.recv(RESPONSE_TIMEOUT)?;
    let tools = tools_response["result"]["tools"]
        .as_array()
        .expect("tools array");
    assert!(!tools.is_empty());
    assert!(
        tools.iter().any(|tool| tool["name"] == "error_metrics"),
        "error_metrics was not listed: {tools:?}"
    );
    for tool in tools {
        assert!(
            tool["inputSchema"].is_object(),
            "missing camelCase inputSchema: {tool}"
        );
        assert!(
            tool.get("input_schema").is_none(),
            "unexpected snake_case input_schema: {tool}"
        );
    }

    server.send(json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "error_metrics",
            "arguments": {
                "enu_covariance_3x3": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 4.0]
                ]
            }
        }
    }))?;
    let metrics = server.recv(RESPONSE_TIMEOUT)?;
    let metrics_result = &metrics["result"];
    assert_eq!(metrics_result["isError"], false);
    assert!(metrics_result["content"].as_array().is_some_and(|items| {
        !items.is_empty() && items[0]["type"] == "text" && items[0]["text"].is_string()
    }));
    assert!(metrics_result["structuredContent"].is_object());
    assert_eq!(metrics_result["structuredContent"]["validity_flag"], true);
    assert!(metrics_result["structuredContent"]["metrics"]["horizontal_radius_m"].is_number());

    server.send(json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}}
    }))?;
    let bad_tool = server.recv(RESPONSE_TIMEOUT)?;
    assert_eq!(bad_tool["id"], 4);
    assert!(bad_tool["result"].is_null());
    assert_eq!(bad_tool["error"]["code"], -32601);
    assert_eq!(bad_tool["error"]["message"], "tool not found");

    server.shutdown()?;
    Ok(())
}

struct McpServer {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<String>,
}

impl McpServer {
    fn spawn() -> TestResult<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_sidereon"))
            .arg("serve-mcp")
            .current_dir(workspace_root())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else {
                    break;
                };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            stdin: Some(stdin),
            rx,
        })
    }

    fn send(&mut self, message: Value) -> TestResult {
        let stdin = self.stdin.as_mut().expect("stdin open");
        serde_json::to_writer(&mut *stdin, &message)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }

    fn recv(&mut self, timeout: Duration) -> TestResult<Value> {
        match self.recv_optional(timeout)? {
            Some(value) => Ok(value),
            None => Err("timed out waiting for MCP response".into()),
        }
    }

    fn recv_optional(&mut self, timeout: Duration) -> TestResult<Option<Value>> {
        match self.rx.recv_timeout(timeout) {
            Ok(line) => {
                let value = serde_json::from_str(&line)?;
                Ok(Some(value))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("MCP stdout closed before response".into())
            }
        }
    }

    fn shutdown(&mut self) -> TestResult {
        drop(self.stdin.take());
        wait_for_child(&mut self.child, RESPONSE_TIMEOUT)
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        let _ = self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> TestResult {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            assert!(status.success(), "unexpected MCP server exit: {status}");
            return Ok(());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return Err("MCP server did not exit after stdin closed".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .to_path_buf()
}
