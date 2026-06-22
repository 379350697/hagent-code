# HAgent Code

Small, self-contained project for running Codex and Claude Code from Hermes
without vendoring the whole Hermes repository.

## Layout

```text
control_plane/                 # platform-neutral /codex and /claude services
hermes_overlay/                # files copied into a Hermes checkout
deploy/                        # install and verification scripts
```

`control_plane/` contains the reusable Codex and Claude command cores.  They
have no Discord or Telegram adapter dependency.

`hermes_overlay/` contains the Hermes integration files that make the control
planes work in a real gateway: app-server / CLI transports, slash-command
integration, Discord native commands, approval bridge, skills, and focused
tests.

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
