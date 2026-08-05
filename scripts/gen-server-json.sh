#!/bin/sh
# POSIX sh on purpose: bash 5.3 (Homebrew's default on macOS, and present on
# GitHub macOS runners) deadlocks on heredocs larger than ~512 bytes.
#
# Generates the server.json submitted to the official MCP registry
# (registry.modelcontextprotocol.io) for a release. Called from the
# mcpb-release workflow with the release version and the SHA-256 of the
# published .mcpb asset. Registry descriptions are limited to 100 chars.
set -eu

VERSION="$1"
SHA256="$2"

cat <<EOF
{
  "\$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.neilberkman/sidereon",
  "title": "Sidereon",
  "description": "GNSS positioning and astrodynamics: orbit propagation, passes, GNSS solves, and RINEX QC",
  "repository": {
    "url": "https://github.com/neilberkman/sidereon",
    "source": "github"
  },
  "websiteUrl": "https://sidereon.dev",
  "version": "$VERSION",
  "packages": [
    {
      "registryType": "mcpb",
      "identifier": "https://github.com/neilberkman/sidereon/releases/download/v${VERSION}/sidereon_${VERSION}.mcpb",
      "fileSha256": "$SHA256",
      "transport": {
        "type": "stdio"
      }
    }
  ]
}
EOF
