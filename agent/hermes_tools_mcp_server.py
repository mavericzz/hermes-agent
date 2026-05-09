"""MCP stdio server that exposes Hermes's ToolRegistry to the local `claude` CLI.

Used by ``claude_local`` mode (see agent/claude_local_adapter.py) so that
Claude Code, when spawned as a subprocess for inference, can call Hermes's
own tools via MCP instead of using its built-in Bash/Read/Edit/etc. The
adapter launches this server with stdio transport and points claude at it
via ``--mcp-config``; ``--allowedTools "mcp__hermes_tools__*"`` plus
``--disallowedTools <claude builtins>`` confines Claude to Hermes's surface.

Architecture & known limitations (Phase 2 MVP):

  Claude spawns this script as a subprocess on its side. That means the
  MCP server runs in a SEPARATE process from the Hermes session that
  invoked it — they re-import the same ToolRegistry but do not share
  in-process state. Concretely:

    Works fine:
      - Disk-backed tools (read_file, write_file, search_files, patch)
      - Bash / terminal / process / execute_code (shell-out tools)
      - Anything that's stateless or mediates state through HERMES_HOME

    Diverges from main hermes session:
      - memory tool (in-process memory cache won't be shared)
      - todo tool (per-session in-memory list)
      - skill_manage / session_search (in-process indices)

  Two pieces of state are propagated explicitly through env vars passed
  by the adapter so the subprocess looks at the same artifacts as the
  parent session:
    - HERMES_HOME (so file paths line up)
    - HERMES_SESSION_ID (so any tool that reads session-scoped files finds them)

  Approval flows: hermes's tools/approval.py prompts via stdin/TUI when a
  sensitive op is attempted. In subprocess mode there is no terminal, so
  approvals will fail closed. The adapter sets HERMES_APPROVAL_MODE=auto
  to grant blanket approval inside the subprocess (caller already opted
  in by selecting claude_local mode).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("hermes.claude_local_mcp")

# Tool name prefix is fixed by the MCP convention claude uses:
# ``mcp__<server-name>__<tool-name>``. Keep this in sync with the
# ``--allowedTools`` glob the adapter passes.
SERVER_NAME = "hermes_tools"


def _ensure_hermes_on_syspath() -> None:
    """Add the hermes project root to sys.path when invoked as a script.

    Claude spawns this script with ``python -m agent.hermes_tools_mcp_server``
    so the parent dir of ``agent`` must be importable. The adapter sets
    ``PYTHONPATH`` to the hermes root before spawning, but be defensive in
    case someone runs this server manually.
    """
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _load_tool_registry():
    """Discover all hermes tool modules so the registry is fully populated."""
    from tools.registry import discover_builtin_tools, registry
    discover_builtin_tools()
    return registry


def _tool_input_schema(entry) -> Dict[str, Any]:
    """Extract the JSON Schema for a tool's input from its registered schema.

    Hermes tool schemas follow OpenAI's function-call shape
    ``{"name", "description", "parameters": {<JSON Schema>}}``.
    The MCP ``Tool.inputSchema`` field expects just the parameters object.
    """
    schema = entry.schema or {}
    params = schema.get("parameters") or schema.get("input_schema") or {}
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    # Some hermes schemas omit the wrapper; coerce to the expected shape.
    if "type" not in params:
        params = {"type": "object", "properties": params.get("properties", {})}
    return params


def _coerce_handler_result(value: Any) -> str:
    """Hermes handlers return JSON strings; tolerate non-string returns too."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


async def _invoke_handler(entry, arguments: Dict[str, Any]) -> str:
    """Call a registered hermes tool handler with MCP-supplied arguments.

    Hermes handlers expect a single positional ``args`` dict. Async handlers
    are awaited; sync handlers are invoked directly. We don't push them into
    a thread pool because most hermes handlers are I/O-bound C extensions or
    short subprocess calls that won't benefit, and the MCP server is single
    -tool-at-a-time anyway (claude waits for one tool result before issuing
    the next).
    """
    handler = entry.handler
    if entry.is_async or inspect.iscoroutinefunction(handler):
        result = await handler(arguments)
    else:
        result = handler(arguments)
        if inspect.iscoroutine(result):
            result = await result
    return _coerce_handler_result(result)


def _build_tool_list(registry) -> List[Any]:
    """Snapshot the registry into a list of MCP Tool objects."""
    from mcp import types as mcp_types

    tools: List[Any] = []
    for entry in registry._snapshot_entries():
        # Skip tools whose check_fn says they're unavailable in this env
        # (e.g. browser tools without playwright installed). The adapter
        # already passes through env vars but installs may not match.
        if entry.check_fn is not None:
            try:
                if not entry.check_fn():
                    continue
            except Exception:
                continue

        description = entry.description or (entry.schema or {}).get("description", "") or entry.name
        tools.append(
            mcp_types.Tool(
                name=entry.name,
                description=description,
                inputSchema=_tool_input_schema(entry),
            )
        )
    return tools


async def _serve() -> None:
    """Run the MCP server over stdio until claude closes the connection."""
    from mcp import types as mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    registry = _load_tool_registry()
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list() -> List[Any]:
        return _build_tool_list(registry)

    @server.call_tool()
    async def _call(name: str, arguments: Dict[str, Any]) -> List[Any]:
        entry = registry.get_entry(name)
        if entry is None:
            return [mcp_types.TextContent(type="text", text=json.dumps({
                "error": f"unknown hermes tool: {name}",
            }))]
        try:
            text = await _invoke_handler(entry, arguments or {})
        except Exception as exc:
            logger.exception("Hermes tool %s raised", name)
            text = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
        return [mcp_types.TextContent(type="text", text=text)]

    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def main() -> None:
    _ensure_hermes_on_syspath()
    # Quiet the noisy hermes loggers — they default to stderr (good) but
    # some emit at INFO which floods the MCP transcript view in claude.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    # MCP uses sys.stdout for its wire protocol. Some hermes module imports
    # print banner messages to stdout at import time; those would corrupt
    # the JSON-RPC stream. Swap stdout to stderr ONLY during the import-
    # heavy serve setup, then restore the real stdout before the MCP loop
    # starts so stdio_server() can speak its protocol.
    real_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        # Pre-import the heavy modules so their banners go to stderr.
        _load_tool_registry()
    finally:
        sys.stdout = real_stdout
    try:
        asyncio.run(_serve())
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
