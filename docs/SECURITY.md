# Security

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/PrayasPanda/polymom/security/advisories/new)
with steps to reproduce and the affected version. You'll get an acknowledgement within
3 working days. Fixes are released as patch versions and credited in the CHANGELOG unless
you prefer otherwise.

Supported versions: the latest `1.x` release.

## Threat model

Meeting recordings and transcripts are sensitive: names, decisions, sometimes personal
data. The main risks are one tenant reading another's meetings, malicious uploads
attacking the media toolchain, the LLM being steered by content spoken in a meeting,
server-side request forgery through webhooks, and leaked secrets.

| Area | Control |
| --- | --- |
| **Authentication** | `X-API-Key` on every non-health route (API and UI partials). Keys are random, shown once, and stored as SHA-256 hashes with a short prefix for identification. Create them with `python -m scripts.create_api_key`; `--list` and `--revoke <prefix>` are also available. `API_KEY_REQUIRED=false` exists for local development only and is refused in production. |
| **Tenant isolation (OWASP API1, BOLA)** | Every meeting records its owning key. Reads, writes, lists, search, exports, audio and duplicate detection are scoped by owner in the repositories, not in the routes. Another tenant's id returns 404. Integration tests (`tests/integration/test_security.py`) cover cross-tenant reads, writes, lists and search. |
| **Rate limiting (API4)** | Per key (per IP when auth is off), with a moving window in Redis that is shared by replicas. Upload, process and regenerate have a stricter limit. 429 responses carry `Retry-After`. |
| **Uploads** | Streamed with a size cap; `Content-Length` is checked before the body is read; empty files are rejected; the extension is allow-listed and the magic bytes must match it; ffprobe must find audio; duration is capped. Filenames are sanitized and never used as storage paths. Blobs live under generated keys outside any served directory. |
| **Media toolchain** | ffmpeg and ffprobe run with `-protocol_whitelist file,pipe,fd` and a timeout, so crafted playlists or containers can't fetch URLs or read other files. |
| **Request bodies** | JSON bodies are capped by `MAX_JSON_BODY_KB` (413 `payload_too_large`); form fields are length-limited. |
| **Webhooks (SSRF, API7)** | Only `http`/`https`. The host is resolved when the URL is submitted and again at delivery; loopback, private, link-local (cloud metadata), reserved and multicast addresses are refused. Payloads are HMAC-SHA256 signed with a timestamp (`X-Polymom-Signature`, `X-Polymom-Timestamp`). |
| **LLM prompt injection** | The transcript is treated as untrusted data inside delimiters. Instruction-like utterances ("ignore previous instructions ...") are detected, flagged in `verification_report.injection_flags` and never followed. Every summary item must quote the transcript, and the verifier drops items whose evidence can't be found. Owners are never guessed. |
| **HTTP headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, HSTS in production. The API's CSP is `default-src 'none'`. The UI has its own CSP with `'self'` only: no inline scripts, no inline styles, no third-party origins. CORS is off unless `CORS_ORIGINS` is set, and `*` is refused in production. |
| **Errors** | Clients get a stable code, a message and a `request_id`. Stack traces and internal details appear only in the logs. |
| **Secrets** | Taken only from the environment or `.env` (git-ignored); `.env.example` has no values. `SecretStr` keeps them out of logs and reprs. With `APP_ENV=production` the app refuses to start without Redis, with auth off, without the LLM and Hugging Face credentials its configured backends need, or with unsafe CORS or webhook settings. |
| **Containers** | Non-root `app` user, slim base pinned by digest, `apt-get upgrade` at build time, no compilers in the runtime image. Trivy fails the image build on fixable CRITICAL vulnerabilities. |
| **Supply chain** | `uv.lock` pins every dependency with hashes. CI runs `pip-audit` on the locked set and `gitleaks` over the full history. Dependabot updates pip, Docker images, compose images and GitHub Actions weekly. |
| **Data retention** | `scripts/cleanup.py` purges raw audio after `RETENTION_DAYS` while keeping results; `DELETE /meetings/{id}` removes everything for a meeting at once. |

## Known accepted risks

- **transformers 4.x advisories** (PYSEC-2025-217, PYSEC-2026-2288/2289/2290/3929) are
  ignored in `pip-audit`. The fixes need transformers ≥ 5.10, which requires
  huggingface-hub ≥ 1.5, and that release removed an argument pyannote.audio 3.x still
  passes. transformers is used only by the optional `indic` extra, to load pinned, trusted
  model repositories (IndicConformer, MMS-LID). The follow-up is a coordinated upgrade to
  pyannote.audio 4 plus transformers 5.
- **Base-image HIGH findings** without a fix are tracked by Trivy's SARIF upload, not
  blocked. At release time the CPU image has 0 CRITICAL and 2 HIGH findings, both in
  setuptools' vendored packages inside the base Python, which the app doesn't use.
- **MMS-LID license** (CC-BY-NC-4.0) is a licensing constraint, not a security issue; see
  [TECHNOLOGY_CHOICES.md](TECHNOLOGY_CHOICES.md#spoken-language-identification).
- The API key is kept in the browser's `localStorage` by the UI. Serve the UI over HTTPS
  only, and prefer short-lived, per-user keys.

## Recommended GitHub settings

Apply these under *Settings → Branches → Branch protection rules* for `main`:

- Require a pull request before merging, with 1 approval, and dismiss stale approvals on
  new commits.
- Require these status checks to pass, with branches up to date: `quality`,
  `dependency-audit`, `secret-scan` (CI), `e2e` (E2E), and `image (cpu)` (Docker).
- Require conversation resolution and a linear history (squash merge).
- Require signed commits (optional, recommended).
- Don't allow bypassing, including for administrators; block force pushes and deletions.
- Restrict who can push version tags `v*` (*Settings → Tags → rulesets*), because tags
  trigger releases and image pushes.

Also enable:

- Dependabot alerts and security updates.
- Secret scanning with push protection.
- Private vulnerability reporting.
- CodeQL default setup (Python).
