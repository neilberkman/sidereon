# Coordinated releases and binding parity

Sidereon releases the Rust core and facade, C, Go, WebAssembly/npm, Python
and Elixir libraries at the same version. A release is complete when every
published package, release tag and native artifact uses that version and
clean consumers have verified the delivered packages.

Every new public API or behavior change must be reviewed across all applicable
interfaces before release. Implement the same domain behavior and error
information through each language's public API, with regression coverage.
Document a language-specific equivalent when an interface delegates that
responsibility to its caller, such as browser-owned HTTP transport.

Prepare manifests, release notes and exact dependency/source identities
together. Publish in dependency order: Rust core and facade, then C and the
bindings that depend on those artifacts. Build Go's native archives from the
matching C release. Verify checksums and source identities against the actual
published artifacts, and update the demo to the matching npm version.

The broader review of existing binding coverage continues separately. A new
change's parity checks belong to its release; the historical survey does not
add unrelated requirements to that release.
