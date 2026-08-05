#!/bin/sh
# POSIX sh on purpose: bash 5.3 (Homebrew's default on macOS, and present on
# GitHub macOS runners) deadlocks on heredocs larger than ~512 bytes.
#
# Assembles the MCPB bundle (https://github.com/anthropics/mcpb) from
# pre-built sidereon-cli binaries. Called from the mcpb-release workflow
# with the release version and a directory containing:
#   sidereon-darwin      (macOS universal: arm64 + x86_64)
#   sidereon-linux       (linux x86_64)
#   sidereon-win32.exe   (windows x86_64)
#
# The bundle carries one binary per OS; the manifest's platform_overrides
# select the right one. MCPB manifests can only discriminate by OS, not
# CPU arch, so linux/windows arm64 users should build from source instead.
set -eu

VERSION="$1"
BINDIR="$2"
OUTDIR="${3:-dist}"

STAGE="$OUTDIR/mcpb-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin"

cp "$BINDIR/sidereon-darwin" "$STAGE/bin/sidereon-darwin"
cp "$BINDIR/sidereon-linux" "$STAGE/bin/sidereon-linux"
cp "$BINDIR/sidereon-win32.exe" "$STAGE/bin/sidereon-win32.exe"
chmod +x "$STAGE/bin/"*

sed "s/__VERSION__/$VERSION/" > "$STAGE/manifest.json" <<'EOF'
{
  "manifest_version": "0.3",
  "name": "sidereon",
  "display_name": "Sidereon",
  "version": "__VERSION__",
  "description": "GNSS positioning and astrodynamics over MCP: orbit propagation, satellite passes, GNSS solves from RINEX, position error metrics, track filtering, and observation QC.",
  "author": {
    "name": "Neil Berkman",
    "email": "neil@xuku.com",
    "url": "https://github.com/neilberkman"
  },
  "homepage": "https://github.com/neilberkman/sidereon",
  "repository": {
    "type": "git",
    "url": "https://github.com/neilberkman/sidereon"
  },
  "license": "MIT",
  "keywords": ["gnss", "astrodynamics", "satellite", "orbit", "rinex"],
  "server": {
    "type": "binary",
    "entry_point": "bin/sidereon-darwin",
    "mcp_config": {
      "command": "${__dirname}/bin/sidereon-darwin",
      "args": ["serve-mcp"],
      "platform_overrides": {
        "linux": {
          "command": "${__dirname}/bin/sidereon-linux"
        },
        "win32": {
          "command": "${__dirname}/bin/sidereon-win32.exe"
        }
      }
    }
  },
  "compatibility": {
    "platforms": ["darwin", "win32", "linux"]
  }
}
EOF

OUT="sidereon_${VERSION}.mcpb"
mkdir -p "$OUTDIR"
(cd "$STAGE" && zip -qr "../$OUT" .)
rm -rf "$STAGE"
echo "built $OUTDIR/$OUT"
