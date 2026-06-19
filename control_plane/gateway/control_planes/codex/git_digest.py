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
        return {"available": False, "error": "未选择工作区"}
    code, root, err = _run_git_command(workspace, ["rev-parse", "--show-toplevel"])
    if code != 0:
        return {"available": False, "error": err or "不是 git 仓库"}
    status_code, status, status_err = _run_git_command(workspace, ["status", "--short"])
    stat_code, diff_stat, stat_err = _run_git_command(workspace, ["diff", "--stat"])
    names_code, names, names_err = _run_git_command(workspace, ["diff", "--name-only"])
    staged_code, staged, _staged_err = _run_git_command(
        workspace,
        ["diff", "--cached", "--name-only"],
    )
    files = [line for line in names.splitlines() if line]
    staged_files = [line for line in staged.splitlines() if line]
    title_subject = "更新 Codex 任务变更"
    if files:
        title_subject = f"更新 {files[0]}" if len(files) == 1 else f"更新 {len(files)} 个文件"
    pr_body_lines = [
        "## 摘要",
        "- 应用 Codex 生成的变更",
        "",
        "## 验证",
        "- Hermes 未自动运行验证",
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
        return f"Git 不可用：{digest.get('error') or '未知错误'}"
    lines = ["Git 摘要"]
    if digest.get("repoRoot"):
        lines.append(f"仓库：{digest['repoRoot']}")
    if digest.get("status"):
        lines.extend(["状态：", str(digest["status"])])
    else:
        lines.append("状态：干净")
    if digest.get("diffStat"):
        lines.extend(["变更统计：", str(digest["diffStat"])])
    files = digest.get("files") if isinstance(digest.get("files"), list) else []
    if files:
        lines.append("文件：" + ", ".join(str(item) for item in files[:12]))
        if len(files) > 12:
            lines.append(f"...另有 {len(files) - 12} 个")
    fallback = "更新 Codex 任务变更"
    lines.append(f"提交标题草稿：{digest.get('commitMessageDraft') or fallback}")
    lines.append(f"PR 标题草稿：{digest.get('prTitleDraft') or fallback}")
    return "\n".join(lines)
