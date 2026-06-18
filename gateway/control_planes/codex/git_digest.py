"""Git summary helpers for Codex control-plane commands."""

from __future__ import annotations

import subprocess


JsonDict = dict[str, object]


def _run_git_command(
    workspace: str,
    args: list[str],
    *,
    timeout: float = 10.0,
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def git_digest(workspace: str) -> JsonDict:
    if not workspace:
        return {"available": False, "error": "workspace is not set"}
    code, root, err = _run_git_command(workspace, ["rev-parse", "--show-toplevel"])
    if code != 0:
        return {"available": False, "error": err or "not a git repository"}
    status_code, status, status_err = _run_git_command(workspace, ["status", "--short"])
    stat_code, diff_stat, stat_err = _run_git_command(workspace, ["diff", "--stat"])
    names_code, names, names_err = _run_git_command(workspace, ["diff", "--name-only"])
    staged_code, staged, _staged_err = _run_git_command(
        workspace,
        ["diff", "--cached", "--name-only"],
    )
    files = [line for line in names.splitlines() if line]
    staged_files = [line for line in staged.splitlines() if line]
    title_subject = "Update Codex task changes"
    if files:
        title_subject = f"Update {files[0]}" if len(files) == 1 else f"Update {len(files)} files"
    pr_body_lines = [
        "## Summary",
        "- Apply Codex-generated changes",
        "",
        "## Validation",
        "- Not run by Hermes automatically",
    ]
    return {
        "available": True,
        "repoRoot": root,
        "status": status if status_code == 0 else "",
        "statusError": status_err if status_code != 0 else "",
        "diffStat": diff_stat if stat_code == 0 else "",
        "diffStatError": stat_err if stat_code != 0 else "",
        "files": files if names_code == 0 else [],
        "filesError": names_err if names_code != 0 else "",
        "stagedFiles": staged_files if staged_code == 0 else [],
        "commitMessageDraft": title_subject,
        "prTitleDraft": title_subject,
        "prBodyDraft": "\n".join(pr_body_lines),
    }


def format_git_digest(digest: JsonDict) -> str:
    if not digest.get("available"):
        return f"Git: unavailable ({digest.get('error') or 'unknown error'})"
    lines = ["Git digest"]
    if digest.get("repoRoot"):
        lines.append(f"Repo: {digest['repoRoot']}")
    if digest.get("status"):
        lines.extend(["Status:", str(digest["status"])])
    else:
        lines.append("Status: clean")
    if digest.get("diffStat"):
        lines.extend(["Diff stat:", str(digest["diffStat"])])
    files = digest.get("files") if isinstance(digest.get("files"), list) else []
    if files:
        lines.append("Files: " + ", ".join(str(item) for item in files[:12]))
        if len(files) > 12:
            lines.append(f"...and {len(files) - 12} more")
    lines.append(f"Commit draft: {digest.get('commitMessageDraft') or 'Update Codex task changes'}")
    lines.append(f"PR title draft: {digest.get('prTitleDraft') or 'Update Codex task changes'}")
    return "\n".join(lines)
