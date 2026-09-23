# sidereon-cli

`sidereon` is a small command-line wrapper over the `sidereon` facade crate. It parses real GNSS products through the library, assembles RINEX observation epochs for SPP through the core convenience API, and prints compact human output or stable JSON where supported.

This branch adds MCP server support with `serve-mcp`.

## Build

From the workspace root:

```sh
cargo build -p sidereon-cli
```

The binary is:

```sh
target/debug/sidereon
```

For a local install into Cargo's bin directory:

```sh
cargo install --path crates/sidereon-cli --locked
```

## Serve MCP over stdio

Run the typed MCP server with a profile filter:

```sh
sidereon serve-mcp --profile gnss
sidereon serve-mcp --profile astro
sidereon serve-mcp --profile all
```

Profiles filter Layer 1 tools and graph nodes:

- `gnss`: solve/log/mask tools plus GNSS graph nodes.
- `astro`: pass-prediction tools plus TLE/time-scan nodes.
- `all`: full graph and all tools.

### Claude Desktop

```json
{
  "mcpServers": {
    "sidereon-serve": {
      "command": "sidereon",
      "args": ["serve-mcp", "--profile", "all"],
      "env": {}
    }
  }
}
```

### Claude Code

```json
{
  "mcpServers": {
    "sidereon-serve": {
      "command": "sidereon",
      "args": ["serve-mcp", "--profile", "gnss"],
      "env": {}
    }
  }
}
```

Configured resources include:

- `sidereon://docs/concepts/frames-time-scales`
- `sidereon://docs/concepts/cep-r95`
- `sidereon://docs/concepts/product-expectations`
- `sidereon://docs/capability-map`

## Commands

Solve RINEX observation epochs with broadcast navigation:

```sh
sidereon solve --obs data/site.obs --nav data/brdc.rnx
sidereon solve --obs data/site.obs --nav data/brdc.rnx --json
```

Use SP3 precise orbits while retaining broadcast context:

```sh
sidereon solve --obs data/site.obs --nav data/brdc.rnx --sp3 data/orbits.sp3
```

Replay a logged OBS/NAV session in a terminal monitor:

```sh
sidereon tui --obs data/site.obs --nav data/brdc.rnx
sidereon tui --obs data/site.obs --nav data/brdc.rnx --speed 20 --paused
```

The TUI replays assembled SPP epochs through the same solve path as `solve`.
It shows the current fix, per-satellite azimuth/elevation and used flag,
recent horizontal scatter, and CEP/R95 bounds.

Watch a live RTCM stream with the same solve pipeline:

```sh
sidereon tui --ntrip https://caster.example.com/ --mount /example --nav data/brdc.rnx
sidereon tui --ntrip https://caster.example.com/ --mount /example --ntrip-nav data/live-nav.rnx --user myname
SIDEREON_NTRIP_PASSWORD=secret sidereon tui --ntrip https://caster.example.com/ --mount /example --nav data/brdc.rnx
sidereon tui --tcp 192.168.0.1:5000 --nav data/brdc.rnx
```

Live mode uses `--user`/`--pass` and prefers `SIDEREON_NTRIP_USER` and
`SIDEREON_NTRIP_PASSWORD` when credentials are omitted. Passwords are never
logged and never shown in status text. You can also send a fixed GGA position to
some casters with:

```sh
sidereon tui --ntrip https://caster.example.com/ --mount /example --nav data/brdc.rnx --gga-lat 37.4220 --gga-lon -122.0838
```

Run RINEX observation lint and observation QC:

```sh
sidereon qc --obs data/site.obs
sidereon qc --obs data/site.obs --json
```

Compute covariance-derived position metrics:

```sh
sidereon metrics --enu-cov "4,0,0,0,9,0,0,0,16"
sidereon metrics --json-file covariance.json --probability 0.99
sidereon metrics --enu-cov "4,0,0,0,9,0,0,0,16" --json
```

Inspect a GNSS file. The format is recognized from its own identifying text
(the RINEX, CRINEX or ANTEX header label, the SP3 `#` version line, or a TLE
line 1 / line 2 pair) and read with that format's parser, so a malformed file
fails with that parser's error. A TLE file keeps every readable element set and
lists each rejected record and checksum mismatch with its line number:

```sh
sidereon inspect data/site.obs
sidereon inspect data/site.crx
sidereon inspect data/brdc.rnx
sidereon inspect data/orbits.sp3
sidereon inspect data/antennas.atx
sidereon inspect data/stations.tle
```

Exit codes are `0` for success, `1` for parse or solve errors, and `2` for command usage errors.
