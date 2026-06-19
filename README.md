# HAgent Code

Small, self-contained project for running Codex from Hermes without vendoring
the whole Hermes repository.

## Layout

```text
control_plane/                 # platform-neutral /codex service
hermes_overlay/                # files copied into a Hermes checkout
deploy/                        # install and verification scripts
```

`control_plane/` contains the reusable Codex command core.  It has no Discord
or Telegram adapter dependency.

`hermes_overlay/` contains the Hermes integration files that make the control
plane work in a real gateway: app-server transport, slash-command integration,
Discord native commands, approval bridge, and focused tests.

Secrets, local state, `.git`, caches, and the rest of Hermes are intentionally
not part of this repository.

## Quick Deploy

Preview what would be installed:

```bash
./deploy/install-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent \
  --dry-run
```

Apply with timestamped backups:

```bash
./deploy/install-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent \
  --apply
```

Verify the target Hermes checkout:

```bash
./deploy/verify-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent
```

See `DEPLOY.md` for details.
