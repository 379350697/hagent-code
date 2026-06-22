# Deploy HAgent Code Into Hermes

This repository is an overlay project.  It keeps the Codex control plane and
the Hermes integration files in one clean Git repository, then installs them
into any Hermes checkout on demand.

## What Gets Installed

The installer copies two source trees into the target Hermes checkout:

- `control_plane/` -> platform-neutral `/codex` command core and tests.
- `hermes_overlay/` -> Hermes-specific transport, approval, slash, Discord,
  API, skill, and test integration files.

The installer does not copy secrets, caches, local databases, `.git`, or the
whole Hermes repository.

## Install

Always preview first:

```bash
./deploy/install-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent \
  --dry-run
```

Apply:

```bash
./deploy/install-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent \
  --apply
```

When applying, the script backs up every existing target file before copying:

```text
$HERMES_AGENT/.hagent-code-backups/YYYYMMDD-HHMMSS/
```

## Verify

Run:

```bash
./deploy/verify-hermes-overlay.sh \
  --hermes-agent /home/wl/.hermes/hermes-agent
```

The verification script checks required files, compiles the Codex control-plane
and overlay modules, then runs the focused Hermes tests when a target virtualenv
exists.

## Runtime Notes

After applying into a running Hermes deployment, restart the gateway:

```bash
systemctl --user restart hermes-gateway.service
```

Then validate from Discord or Telegram:

```text
/codex doctor
/codex status
/codex continue <task>

/claude doctor
/claude status
/claude continue <task>
```

Discord slash commands may require a client refresh or guild command sync before
new autocomplete metadata is visible.
