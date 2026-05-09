"""Claude Code CLI adapter for Hermes Agent.

Routes Anthropic-Messages-shaped API calls through the locally-installed
``claude`` binary instead of api.anthropic.com directly.

Why this exists:
    A third-party tool calling api.anthropic.com with a Claude Max OAuth token
    sends ``user-agent: claude-cli/<v> (external, cli)`` and gets metered
    against the small "extra usage" bucket Anthropic carves out of Max plans
    for third-party OAuth tools. That bucket exhausts quickly.

    The official ``claude`` binary identifies as itself and draws from the
    main Max bucket. Spawning ``claude --print`` as a subprocess routes the
    HTTP call through it, so usage bills against Max instead of the third
    party bucket. This is the same workaround Paperclip uses
    (paperclip/packages/adapters/claude-local).

Tools (Phase 2 — MCP bridge):
    Hermes's tool registry is exposed to the spawned ``claude`` via an MCP
    stdio server (``agent.hermes_tools_mcp_server``). The adapter writes a
    temp ``--mcp-config`` JSON pointing at it and passes
    ``--allowedTools "mcp__hermes_tools__*"`` so Claude can ONLY call
    hermes's tools — its built-in Bash/Read/Edit/Glob/Grep/etc. are off.

    Each ``claude --print`` invocation spawns its own MCP subprocess, which
    re-imports the ToolRegistry. Disk-backed tools (read_file, write_file,
    search_files, patch, terminal, execute_code) work transparently because
    their state IS the filesystem. Tools that depend on hermes in-process
    state (memory, todo, session_search) will run against an empty cache
    inside the subprocess — that divergence is the documented Phase-2 gap.

    The adapter propagates ``HERMES_HOME`` and ``HERMES_SESSION_ID`` to the
    subprocess so any HERMES_HOME-rooted artifacts (sessions, memory files,
    skills) line up. Approvals run in ``auto`` mode inside the subprocess
    since there's no terminal to prompt.

Wire format:
    ``claude --print -- --output-format stream-json --verbose`` emits one
    JSONL event per line. The events we care about:

      {"type":"assistant","message":{...Anthropic Message...}, ...}
      {"type":"result","subtype":"success","result":"...","stop_reason":"end_turn",
       "usage":{...},"total_cost_usd":...}

    The ``message`` field on assistant events IS already a fully-shaped
    Anthropic Messages API response. We collect them, prefer the last
    assistant block as the response object, and fold the ``result`` event's
    aggregated ``usage`` and ``stop_reason`` over it so token counts
    reflect the entire turn (not just the last sub-message).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default upper bound on sub-turns per claude invocation. Each "turn" inside
# claude is one assistant message + (optionally) one tool round-trip. Keeping
# this high lets claude finish multi-step work in a single hermes turn; lower
# it via env var if you want tighter control.
_DEFAULT_MAX_TURNS = int(os.environ.get("HERMES_CLAUDE_LOCAL_MAX_TURNS", "50"))

# Per-call timeout. Claude can take a while when running tools internally.
_DEFAULT_TIMEOUT_SEC = int(os.environ.get("HERMES_CLAUDE_LOCAL_TIMEOUT_SEC", "1200"))

# MCP server name MUST match agent/hermes_tools_mcp_server.py:SERVER_NAME so
# the --allowedTools glob below resolves. Don't change one without the other.
_MCP_SERVER_NAME = "hermes_tools"
_MCP_TOOL_GLOB = f"mcp__{_MCP_SERVER_NAME}__*"

# Set HERMES_CLAUDE_LOCAL_DISABLE_TOOLS=1 to fall back to Phase 1 behavior
# (no MCP bridge, claude uses its own tools). Useful as an escape hatch if
# the MCP server is broken or you want Claude's native toolset for a
# specific session.
_DISABLE_HERMES_TOOLS = os.environ.get("HERMES_CLAUDE_LOCAL_DISABLE_TOOLS", "").strip() in ("1", "true", "yes")


class ClaudeLocalError(RuntimeError):
    """Raised when the claude subprocess fails or emits an unparseable transcript."""


def _resolve_claude_binary() -> str:
    """Locate the ``claude`` binary, raising if not on PATH."""
    path = shutil.which("claude")
    if not path:
        raise ClaudeLocalError(
            "The 'claude' CLI is not installed. Install it with: "
            "npm install -g @anthropic-ai/claude-code"
        )
    return path


def _render_messages_as_prompt(
    messages: List[Dict[str, Any]],
    system: Optional[Any] = None,
) -> str:
    """Flatten Anthropic-format messages + system blocks into one prompt string.

    Hermes accumulates conversation state on its side and passes the full
    history each call. ``claude --print`` only takes a single prompt over
    stdin, so we render the whole thing as a transcript with ROLE: prefixes.
    Claude tolerates this fine — it's just text in its context window.

    System blocks are emitted as a leading SYSTEM: section. Tool definitions
    in ``messages`` (tool_use / tool_result blocks from prior turns) are
    rendered too so the conversation reads coherently, even though claude
    will pick its own tools going forward.
    """
    sections: List[str] = []

    if system:
        sys_text = _stringify_content(system)
        if sys_text:
            sections.append(f"SYSTEM:\n{sys_text}")

    # Single-message case: send the body verbatim, no role label. Role
    # prefixes ("USER:") confuse claude when the message body itself contains
    # role-shaped content (e.g. title-generation prompts that embed
    # "User: ...\n\nAssistant: ..." literally), making it treat the outer
    # label as a template placeholder and return "(response)".
    actionable = [m for m in messages if isinstance(m, dict) and _stringify_content(m.get("content"))]
    if not system and len(actionable) == 1:
        return _stringify_content(actionable[0].get("content")).strip()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").upper()
        body = _stringify_content(msg.get("content"))
        if body:
            sections.append(f"{role}:\n{body}")

    return "\n\n".join(sections).strip()


def _stringify_content(content: Any) -> str:
    """Render an Anthropic content payload (str | list of blocks | dict) as text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _stringify_content(content.get("text") or content.get("content"))
    if not isinstance(content, list):
        return str(content)

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Surface prior reasoning so claude has the same context the
            # original model produced. Wrapped so the new turn doesn't
            # confuse it for instructions.
            t = block.get("thinking", "")
            if t:
                parts.append(f"<prior-reasoning>\n{t}\n</prior-reasoning>")
        elif btype == "tool_use":
            name = block.get("name", "?")
            input_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"<tool-call name={name!r}>{input_json}</tool-call>")
        elif btype == "tool_result":
            result = _stringify_content(block.get("content"))
            parts.append(f"<tool-result>{result}</tool-result>")
        elif btype == "image":
            parts.append("[image omitted in claude_local mode]")
    return "\n".join(p for p in parts if p)


def _hermes_root() -> str:
    """Return the absolute path to the hermes-agent project root.

    Used to set PYTHONPATH for the MCP subprocess so it can import
    ``agent.hermes_tools_mcp_server`` and the ``tools`` package.
    """
    # agent/claude_local_adapter.py → parent.parent = repo root.
    return str(Path(__file__).resolve().parent.parent)


def _write_mcp_config(extra_env: Dict[str, str]) -> str:
    """Write a temp ``--mcp-config`` JSON pointing at the hermes-tools server.

    Claude reads this and spawns the named server as a stdio subprocess for
    the duration of the ``--print`` invocation. The file is left on disk so
    claude can re-open it if it restarts the server mid-session; the OS will
    clean ``/tmp`` on reboot. Hermes restarts get fresh paths since the file
    name is randomized.
    """
    cfg = {
        "mcpServers": {
            _MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", "agent.hermes_tools_mcp_server"],
                "env": extra_env,
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix="hermes_claude_local_mcp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


def _build_argv(
    binary: str,
    *,
    model: Optional[str],
    max_turns: int,
    mcp_config_path: Optional[str],
    system_prompt: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    argv = [
        binary,
        "--print",
        "-",
        "--output-format", "stream-json",
        "--input-format", "text",
        "--verbose",
        "--max-turns", str(max_turns),
        # Skip user-level settings (~/.claude/settings.json). Those carry the
        # interactive user's hooks, plugins, and skill configurations — none
        # of which apply when claude is being driven by hermes. Letting them
        # load injects out-of-band context (e.g. SessionStart hooks that
        # prepend persistent behavioral rules) and burns cache tokens on
        # content the model doesn't need. Project + local settings still
        # load so per-repo configuration is honored.
        "--setting-sources", "project,local",
    ]
    if model:
        argv.extend(["--model", model])
    if system_prompt:
        # Replace claude's default system prompt entirely. This bypasses
        # CLAUDE.md auto-discovery, auto-memory injection, dynamic
        # cwd/env/git-status sections, and any user-level skills loaded
        # via SessionStart hooks. Hermes already builds the full system
        # prompt — we don't want claude to overlay its own context on top
        # (which contaminates auxiliary tasks like title generation with
        # the user's terminal-mode behavior rules).
        argv.extend(["--system-prompt", system_prompt])
    if mcp_config_path:
        # Restrict claude to ONLY the hermes_tools MCP server. Built-in
        # tools (Bash/Read/Edit/Glob/Grep/Task/WebFetch/...) get filtered
        # out because they don't match the allowedTools glob.
        argv.extend([
            "--mcp-config", mcp_config_path,
            "--allowedTools", _MCP_TOOL_GLOB,
            # No interactive terminal in subprocess mode, so any MCP tool
            # that triggers approval would hang forever. Skip permission
            # prompts — the user opted in by selecting claude_local mode.
            "--dangerously-skip-permissions",
        ])
    else:
        # No MCP bridge — auxiliary tasks (title gen, compression). Disable
        # claude's built-in tools so it doesn't waste turns trying to read
        # files or run shell commands. Empty string = no tools allowed.
        argv.extend(["--tools", ""])
    if extra_args:
        argv.extend(extra_args)
    return argv


def _drain_stderr(stream, sink: List[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            sink.append(line)
    except Exception:
        pass


def _spawn_and_iter(
    argv: List[str],
    prompt: str,
    *,
    cwd: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
):
    """Spawn ``claude --print`` and yield parsed JSONL events as they arrive.

    This is the streaming primitive shared by both the blocking
    ``_spawn_and_collect`` (Phase 1/2 non-streaming path) and the
    ``messages.stream()`` context manager (Phase 3 streaming path).

    The generator owns the subprocess lifecycle: it spawns, writes the
    prompt to stdin, yields one event dict per JSONL line as it appears
    on stdout, and on exhaustion (or exception) ensures the process is
    reaped and stderr captured. Callers don't need to know about
    subprocess plumbing.

    Yields:
        Dict[str, Any] — one parsed JSONL event per yield.

    Raises:
        ClaudeLocalError on subprocess failure (non-zero exit, timeout,
        or zero parseable events). These are raised AFTER the generator
        has been exhausted; mid-stream subprocess errors are surfaced
        when the generator's __next__ runs out of stdout to read.
    """
    env = dict(os.environ)
    # Strip any third-party identifier the parent process may have set so the
    # subprocess presents itself as plain claude-cli to Anthropic. This is
    # already the default but defensive.
    for k in ("HERMES_USER_AGENT", "ANTHROPIC_USER_AGENT"):
        env.pop(k, None)
    # Signal to user-level hooks that this is hermes spawning claude for
    # inference, not an interactive session. Hooks that inject persistent
    # behavioral context (caveman mode, persistent system reminders, etc.)
    # must short-circuit on this flag — otherwise that context leaks into
    # auxiliary tasks (title gen, compression) and hijacks the response.
    env["HERMES_CLAUDE_SUBPROCESS"] = "1"

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        text=True,
    )

    stderr_lines: List[str] = []
    err_thread = threading.Thread(
        target=_drain_stderr, args=(proc.stderr, stderr_lines), daemon=True
    )
    err_thread.start()

    parse_failures: List[str] = []
    yielded = 0
    try:
        try:
            proc.stdin.write(prompt)
            proc.stdin.flush()
            proc.stdin.close()
        except BrokenPipeError as exc:
            proc.kill()
            raise ClaudeLocalError(f"claude rejected stdin: {exc}") from exc

        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_failures.append(f"{exc}: {line[:200]}")
                continue
            yielded += 1
            yield event
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            exit_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = -1
        err_thread.join(timeout=2)

        if exit_code == -1:
            raise ClaudeLocalError(
                f"claude subprocess timed out after {timeout_sec}s"
            )
        if exit_code != 0:
            tail = "".join(stderr_lines[-40:]).strip()
            raise ClaudeLocalError(
                f"claude exited with status {exit_code}: {tail or '<no stderr>'}"
            )
        if parse_failures and yielded == 0:
            raise ClaudeLocalError(
                "claude produced no parseable output; first errors: "
                + " | ".join(parse_failures[:3])
            )


def _spawn_and_collect(
    argv: List[str],
    prompt: str,
    *,
    cwd: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> List[Dict[str, Any]]:
    """Run claude --print, pipe the prompt to stdin, return parsed JSONL events.

    Thin wrapper over :func:`_spawn_and_iter` for callers that want the
    full transcript rather than streaming visibility.
    """
    return list(_spawn_and_iter(argv, prompt, cwd=cwd, timeout_sec=timeout_sec))


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace so attribute access works.

    Hermes consumers (e.g. ``transports/anthropic.py``) read ``response.content``,
    ``block.type``, ``block.text``, etc. as attributes. Lists stay as lists.
    """
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _build_response(events: List[Dict[str, Any]]) -> Any:
    """Assemble a single Anthropic-Messages-shaped response from stream events.

    Strategy:
      - Concatenate text from every ``assistant`` event in order so the
        caller sees the full reply, not just the final sub-turn.
      - Carry through any tool_use blocks emitted by the LAST assistant
        event (they represent claude's final outstanding action — but in
        practice claude_local never returns with pending tool_use because
        it executes its own tools internally before stopping).
      - Pull aggregate ``usage`` and ``stop_reason`` from the ``result``
        event when present (they reflect the whole turn).
    """
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    result_event = next(
        (e for e in reversed(events) if e.get("type") == "result"),
        None,
    )

    if not assistant_events:
        # Possible if claude failed silently or only emitted system events.
        # Fall back to the result event's text if we have one.
        if result_event and result_event.get("result"):
            text = result_event["result"]
            stop_reason = result_event.get("stop_reason") or "end_turn"
        else:
            # No assistant content. Surface every diagnostic field we
            # can find from the rate_limit_event and result event so the
            # caller (and hermes' error classifier) can tell rate-limit
            # apart from other early-exit causes (auth, MCP startup
            # failure, prompt rejection, etc.). Avoid claiming "rate
            # limit" unless the data actually says so — `allowed_warning`
            # is just a heads-up, not a block.
            rate_limit_event = next(
                (e for e in events if e.get("type") == "rate_limit_event"),
                None,
            )
            rl_info = (rate_limit_event or {}).get("rate_limit_info", {}) or {}
            rl_status = str(rl_info.get("status") or "").lower()
            rl_util = rl_info.get("utilization")
            rl_resets_at = rl_info.get("resetsAt") or rl_info.get("resets_at")
            rl_kind = rl_info.get("rateLimitType") or rl_info.get("rate_limit_type") or ""

            result_subtype = str((result_event or {}).get("subtype") or "")
            result_is_error = bool((result_event or {}).get("is_error"))
            result_text = str((result_event or {}).get("result") or "")
            api_error_status = (result_event or {}).get("api_error_status")

            BLOCKED_STATUSES = {"blocked", "exceeded", "denied"}
            is_actually_blocked = (
                rl_status in BLOCKED_STATUSES
                or (isinstance(rl_util, (int, float)) and rl_util >= 1.0)
            )

            if is_actually_blocked:
                parts = ["Claude Code rate limit reached"]
                if rl_kind:
                    parts.append(f"({rl_kind})")
                if isinstance(rl_util, (int, float)):
                    parts.append(f"— utilization {rl_util * 100:.0f}%")
                if rl_resets_at:
                    try:
                        import datetime as _dt
                        ts = _dt.datetime.fromtimestamp(
                            int(rl_resets_at), tz=_dt.timezone.utc
                        )
                        parts.append(f"— resets {ts.isoformat()}")
                    except Exception:
                        parts.append(f"— resets at {rl_resets_at}")
                msg = " ".join(parts) + (
                    ". Your Claude Max plan quota is exhausted; "
                    "wait for the reset, switch provider with `hermes model`, "
                    "or set HERMES_INFERENCE_PROVIDER=anthropic with an API key."
                )
                raise ClaudeLocalError(msg)

            # Generic empty-response error — include whatever diagnostic
            # detail the events expose so the cause isn't a guess.
            diag_bits = []
            diag_bits.append(
                f"events={[e.get('type') for e in events]}"
            )
            if result_subtype:
                diag_bits.append(f"result.subtype={result_subtype}")
            if result_is_error:
                diag_bits.append("result.is_error=true")
            if api_error_status:
                diag_bits.append(f"result.api_error_status={api_error_status}")
            if result_text:
                diag_bits.append(f"result.text={result_text[:300]!r}")
            if rl_status:
                diag_bits.append(f"rate_limit.status={rl_status}")
            if isinstance(rl_util, (int, float)):
                diag_bits.append(f"rate_limit.utilization={rl_util:.2f}")

            raise ClaudeLocalError(
                "claude produced no assistant message — " + " ".join(diag_bits)
            )
        message_dict = {
            "id": result_event.get("session_id", "msg_local"),
            "type": "message",
            "role": "assistant",
            "model": "claude-local",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _normalize_usage(result_event.get("usage", {})),
        }
        return _to_namespace(message_dict)

    # claude_local resolves tool_use → tool_result internally before
    # stopping, so intermediate assistant events contain tool_use blocks
    # that are already satisfied by subsequent "user" tool_result events
    # in the same stream. Surfacing those upstream causes hermes to
    # re-dispatch tools it never originated, eating the final reply.
    # Keep only text blocks from intermediate events; preserve the LAST
    # event's content verbatim (it carries the model's final output and
    # any genuinely-pending tool_use).
    merged_content: List[Dict[str, Any]] = []
    last_message: Dict[str, Any] = assistant_events[-1].get("message") or {}
    for ev in assistant_events[:-1]:
        msg = ev.get("message") or {}
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                merged_content.append(block)
    for block in last_message.get("content", []) or []:
        if isinstance(block, dict):
            merged_content.append(block)

    # Prefer the result event's stop_reason and usage (whole-turn aggregate).
    # Fall back to the last assistant message's values if no result event.
    if result_event:
        stop_reason = result_event.get("stop_reason") or "end_turn"
        usage_raw = result_event.get("usage", {}) or {}
    else:
        stop_reason = last_message.get("stop_reason") or "end_turn"
        usage_raw = last_message.get("usage", {}) or {}

    message_dict = {
        "id": last_message.get("id") or "msg_local",
        "type": "message",
        "role": "assistant",
        "model": last_message.get("model") or "claude-local",
        "content": merged_content or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": last_message.get("stop_sequence"),
        "usage": _normalize_usage(usage_raw),
    }
    return _to_namespace(message_dict)


def _normalize_usage(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce stream-json usage dict into the shape the Anthropic SDK exposes."""
    if not isinstance(raw, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    return {
        "input_tokens": int(raw.get("input_tokens", 0) or 0),
        "output_tokens": int(raw.get("output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(raw.get("cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(raw.get("cache_read_input_tokens", 0) or 0),
        "service_tier": raw.get("service_tier"),
    }


# ─── Streaming bridge: claude stream-json → Anthropic SDK stream events ──────


def _translate_assistant_event_to_sdk_events(claude_event: Dict[str, Any], block_index_start: int):
    """Yield Anthropic-SDK-shaped stream events for one ``assistant`` claude event.

    The hermes streaming consumer in run_agent.py expects a sequence like:

        content_block_start  (block_index=0, content_block.type="text")
        content_block_delta  (delta.type="text_delta", delta.text="...")
        content_block_stop   (block_index=0)
        content_block_start  (block_index=1, content_block.type="tool_use", name=...)
        content_block_stop   (block_index=1)

    Claude emits one ``assistant`` event per sub-turn carrying the FULL
    content list for that sub-turn (already buffered server-side).  We
    can't deliver per-character text deltas — claude has already rolled
    them up — but per-block events are enough for hermes to:
      - call _fire_first_delta() so the spinner stops
      - call _fire_stream_delta() so the user sees text as soon as a
        sub-turn completes (rather than after the whole turn)
      - call _fire_tool_gen_started() when claude starts a tool

    Yields ``SimpleNamespace`` objects instead of real Anthropic SDK
    types so we don't have to depend on the SDK's internal class layout.
    """
    msg = claude_event.get("message") or {}
    content_blocks = msg.get("content") or []

    next_index = block_index_start
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "") or ""
            yield SimpleNamespace(
                type="content_block_start",
                index=next_index,
                content_block=SimpleNamespace(type="text", text=""),
            )
            if text:
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=next_index,
                    delta=SimpleNamespace(type="text_delta", text=text),
                )
            yield SimpleNamespace(
                type="content_block_stop",
                index=next_index,
            )
            next_index += 1

        elif block_type == "thinking":
            thinking = block.get("thinking", "") or ""
            yield SimpleNamespace(
                type="content_block_start",
                index=next_index,
                content_block=SimpleNamespace(type="thinking", thinking=""),
            )
            if thinking:
                yield SimpleNamespace(
                    type="content_block_delta",
                    index=next_index,
                    delta=SimpleNamespace(type="thinking_delta", thinking=thinking),
                )
            yield SimpleNamespace(
                type="content_block_stop",
                index=next_index,
            )
            next_index += 1

        elif block_type == "tool_use":
            yield SimpleNamespace(
                type="content_block_start",
                index=next_index,
                content_block=SimpleNamespace(
                    type="tool_use",
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ),
            )
            yield SimpleNamespace(
                type="content_block_stop",
                index=next_index,
            )
            next_index += 1

    return next_index


class _ClaudeLocalStream:
    """Context manager mimicking ``anthropic.Anthropic().messages.stream(...)``.

    Hermes's streaming consumer (``run_agent.py:_call_anthropic`` near line
    7054) iterates the context-manager value as an event stream and finally
    calls ``stream.get_final_message()`` for downstream tool extraction.
    We expose the same surface so the existing path works unchanged.

    Implementation: spawn ``claude --print`` once, iterate stream-json
    events as they arrive, translate each ``assistant`` event to the
    Anthropic SDK's content_block_* event shape, accumulate a final
    Message object as we go.
    """

    def __init__(
        self,
        client: "ClaudeLocalClient",
        api_kwargs: Dict[str, Any],
    ):
        self._client = client
        self._api_kwargs = api_kwargs
        self._iter_gen = None
        self._final_events: List[Dict[str, Any]] = []
        self._mcp_config_path: Optional[str] = None

    def __enter__(self) -> "_ClaudeLocalStream":
        argv, mcp_path, prompt = self._client._build_invocation(self._api_kwargs)
        self._mcp_config_path = mcp_path
        self._iter_gen = _spawn_and_iter(
            argv,
            prompt,
            cwd=self._client._cwd,
            timeout_sec=self._client._timeout_sec,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Drain the generator if the consumer broke out early so the
        # subprocess gets reaped via its finally block. close() runs the
        # generator's finally clauses synchronously.
        gen = self._iter_gen
        self._iter_gen = None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        if self._mcp_config_path:
            try:
                os.unlink(self._mcp_config_path)
            except OSError:
                pass
            self._mcp_config_path = None

    def __iter__(self):
        if self._iter_gen is None:
            raise ClaudeLocalError("stream not entered")
        block_index = 0
        for raw in self._iter_gen:
            etype = raw.get("type")
            if etype == "assistant":
                self._final_events.append(raw)
                for sdk_event in _translate_assistant_event_to_sdk_events(raw, block_index):
                    if isinstance(sdk_event, int):
                        block_index = sdk_event
                    else:
                        yield sdk_event
                # _translate is a generator — we won't get its return
                # value via yield. Re-derive next index from the event
                # we just emitted by counting content_block_stop events
                # in the assistant message.
                msg_blocks = (raw.get("message") or {}).get("content") or []
                block_index += sum(
                    1 for b in msg_blocks
                    if isinstance(b, dict) and b.get("type") in ("text", "thinking", "tool_use")
                )
            elif etype == "result":
                self._final_events.append(raw)
                # No SDK-side event to emit — the hermes consumer detects
                # end-of-stream via generator exhaustion.
            elif etype in ("system", "user", "rate_limit_event"):
                # Carry through but don't translate; hermes only reacts to
                # assistant content blocks.
                self._final_events.append(raw)

    def get_final_message(self) -> Any:
        """Return the merged Anthropic-Messages-shaped response.

        Same shape as the non-streaming path produces, built from the
        events accumulated during iteration.  Safe to call after
        iteration completes (or after early break — partial events still
        yield a usable message).
        """
        return _build_response(self._final_events)


# ─── Public client surface ────────────────────────────────────────────────────


class _ClaudeLocalMessagesNamespace:
    """Mimics ``anthropic.Anthropic().messages`` so existing call sites work unchanged."""

    def __init__(self, client: "ClaudeLocalClient"):
        self._client = client

    def create(self, **api_kwargs) -> Any:
        return self._client._invoke(api_kwargs)

    def stream(self, **api_kwargs) -> _ClaudeLocalStream:
        return _ClaudeLocalStream(self._client, api_kwargs)


class ClaudeLocalClient:
    """Drop-in replacement for ``anthropic.Anthropic`` that spawns ``claude --print``.

    Only the subset of the SDK that hermes actually uses is implemented:
      - ``.messages.create(**kwargs)`` returning a response with ``.content``,
        ``.usage``, ``.stop_reason``, etc. attribute-accessible.
      - ``.close()`` no-op so the existing client-rebuild path is harmless.

    Streaming (``messages.stream(...)``) is NOT yet implemented; hermes's
    non-streaming path is sufficient for the MVP. If a streaming code path
    is hit, we raise so the caller can fall back to non-streaming.
    """

    def __init__(
        self,
        *,
        max_turns: int = _DEFAULT_MAX_TURNS,
        timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
        cwd: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self._binary = _resolve_claude_binary()
        self._max_turns = max_turns
        self._timeout_sec = timeout_sec
        self._cwd = cwd
        self._extra_args = list(extra_args or [])
        self.messages = _ClaudeLocalMessagesNamespace(self)

    def _build_invocation(self, api_kwargs: Dict[str, Any]):
        """Translate Anthropic-SDK kwargs into a (argv, mcp_path, prompt) tuple.

        Shared by ``_invoke`` (non-streaming) and the streaming context
        manager so both paths produce identical claude invocations and
        hermes tool exposure.

        Returns:
            Tuple of (argv list, mcp_config_path | None, prompt string).
            Caller is responsible for unlinking ``mcp_config_path`` when
            done, even on exception.
        """
        model = api_kwargs.get("model")
        messages = api_kwargs.get("messages") or []
        system = api_kwargs.get("system")
        tools = api_kwargs.get("tools") or []

        if tools:
            logger.debug(
                "claude_local: %d hermes tool schemas dropped from prompt; "
                "they are exposed to claude via MCP instead",
                len(tools),
            )

        # Render the message history without folding `system` into the prompt
        # body — system goes via `--system-prompt` so claude doesn't also pull
        # in CLAUDE.md / auto-memory / hook context on top of it.
        prompt = _render_messages_as_prompt(messages, system=None)
        system_prompt_text = _stringify_content(system) if system else None
        if not prompt:
            # Fall back to the system text as the prompt body when there are
            # no messages — claude --print requires a non-empty stdin.
            if system_prompt_text:
                prompt = system_prompt_text
                system_prompt_text = None
            else:
                raise ClaudeLocalError("claude_local: empty prompt after rendering messages")

        mcp_config_path: Optional[str] = None
        # Only attach the hermes-tools MCP server when the caller actually
        # requested tools. Auxiliary tasks (title generation, compression,
        # session search) pass `tools=None` and don't need the MCP bridge —
        # spawning it for them adds latency and exposes a startup race
        # where claude can return "result: success" with no assistant
        # message if MCP init churns long enough.
        if tools and not _DISABLE_HERMES_TOOLS:
            subprocess_extra_env: Dict[str, str] = {}
            for var in ("HERMES_HOME", "HERMES_SESSION_ID", "PYTHONPATH"):
                v = os.environ.get(var)
                if v:
                    subprocess_extra_env[var] = v
            existing_pp = subprocess_extra_env.get("PYTHONPATH", "")
            hermes_root = _hermes_root()
            if hermes_root not in existing_pp.split(os.pathsep):
                subprocess_extra_env["PYTHONPATH"] = (
                    f"{hermes_root}{os.pathsep}{existing_pp}".rstrip(os.pathsep)
                )
            subprocess_extra_env["HERMES_APPROVAL_MODE"] = os.environ.get(
                "HERMES_APPROVAL_MODE", "auto"
            )
            try:
                mcp_config_path = _write_mcp_config(subprocess_extra_env)
            except OSError as exc:
                logger.warning(
                    "claude_local: could not write MCP config (%s); "
                    "falling back to claude's built-in tools", exc,
                )

        argv = _build_argv(
            self._binary,
            model=model,
            max_turns=self._max_turns,
            mcp_config_path=mcp_config_path,
            system_prompt=system_prompt_text,
            extra_args=self._extra_args,
        )
        logger.debug("claude_local: spawning %s", " ".join(argv))
        return argv, mcp_config_path, prompt

    def _invoke(self, api_kwargs: Dict[str, Any]) -> Any:
        argv, mcp_config_path, prompt = self._build_invocation(api_kwargs)
        try:
            events = _spawn_and_collect(
                argv,
                prompt,
                cwd=self._cwd,
                timeout_sec=self._timeout_sec,
            )
        finally:
            if mcp_config_path:
                try:
                    os.unlink(mcp_config_path)
                except OSError:
                    pass
        return _build_response(events)

    def close(self) -> None:
        """No-op — the subprocess is per-call, no persistent state to release."""
        return None


def build_claude_local_client(**kwargs) -> ClaudeLocalClient:
    """Factory mirroring ``build_anthropic_client``'s call signature shape."""
    return ClaudeLocalClient(**kwargs)
