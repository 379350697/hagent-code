# HAgent Codex Control Plane

Small, platform-neutral Codex control plane for Hermes gateway deployments.

The code lives under `gateway/control_planes/codex/` so it stays physically
separate from Hermes platform adapters and gateway infrastructure. Discord and
Telegram should remain independent adapters; both should delegate `/codex`
commands to this control plane through `CommandRequest`.

See `DEPLOY.md` for installation steps.
