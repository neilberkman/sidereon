# OTP 30 removes `:ftp`: replacement plan for the Elixir FTP transport

Status: IMPLEMENTED (2026-08-04). `Sidereon.GNSS.FtpClient` - a minimal
passive-mode anonymous-FTP client over `:gen_tcp` - ships in the Elixir
interface and is the default for the `:ftp_module` seam, removing the
dependency on the deprecated application entirely. Verified against an
in-process fake FTP server and live against the WHU archive (listing,
exact acquisition, merge). CI carries an OTP 29 leg. Remaining from the
plan below: add an OTP 30 leg when the first RC ships (~May 2027).

Original problem statement: the Elixir package declares
`elixir: "~> 1.18"` with no OTP ceiling, so the day OTP 30 ships, any user
adopting it would have gotten `:undef` at runtime on the FTP acquisition
path (the WHU `wum_nrt` line and FTP publication-status listings).

## Verified facts (2026-08-04)

- OTP 30 removes `ftp:_/_` (and `ct_ftp`, `mod_cgi`, `odbc`, archive
  functionality, legacy TLS versions):
  https://www.erlang.org/doc/scheduled_for_removal.html
- The entire official migration guidance is "use SFTP" - inapplicable to a
  public anonymous-FTP archive we do not operate. WHU's IGS data center
  (`igs.gnsswhu.cn`) serves FTP only; no HTTP surface exists (verified
  live).
- OTP is NOT splitting the application out: no `ftp` package exists on
  hex.pm, no `erlang/ftp` standalone repository exists, and the hex
  landscape has no maintained general-purpose BEAM FTP client (SFTP
  clients, FTP servers, one Gleam client).

## Why we are structurally ready

The 0.36.1+ transport dispatches every FTP call through a runtime-resolved
module: `Application.get_env(:sidereon, :ftp_module, :ftp)`. Swapping the
default module requires zero call-site changes, and a user can override it
today. The surface the transport uses is deliberately tiny: `open/2`
(passive), `user/3` (anonymous), `type/2` (binary), `ls/2`,
`recv_chunk_start/2`, `recv_chunk/1`, `close/1`.

## Plan

1. **Ship `Sidereon.GNSS.FtpClient` before OTP 30**: a minimal client over
   `:gen_tcp` implementing exactly the surface above - anonymous login,
   `TYPE I`, passive-mode `RETR` and `LIST` - with the same bounded
   semantics the transport already enforces (connect timeout, streamed
   chunk cap, 550 as archive absence). A few hundred lines, fully testable
   against the existing `:ftp_client` injection seam plus the
   network-tagged live WHU test. Preferred over vendoring OTP's full ftp
   application (active mode, FTPS, and service supervision we never use);
   vendoring (Apache-2.0, `lib/ftp` in erlang/otp) remains the fallback if
   protocol edge cases surface against real archives.
2. **Make it the default** once it passes the live gates, dropping the
   dependency on deprecated OTP code entirely; keep `:ftp_module` as the
   escape hatch.
3. **CI matrix**: add an OTP 29 leg now (29-only deprecation behavior was
   caught locally, never in CI) and an OTP 30 leg at first RC. The absence
   of an OTP ceiling in `mix.exs` is a compatibility claim; CI should test
   the newest OTP that claim covers.

## Trigger

Implement at the next Elixir release cycle or when OTP 30 reaches RC,
whichever comes first. The WHU line is load-bearing for downstream merge
sets as of 0.36.3; this transport cannot be allowed to lapse.
