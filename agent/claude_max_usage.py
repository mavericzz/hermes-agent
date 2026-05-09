"""Claude Max plan usage telemetry, sourced from local session logs via ccusage.

Background:
    Anthropic does not expose a public API endpoint for Max-plan weekly /
    block utilisation that hermes can call without scraping the OAuth token
    out of the user's keychain.  The community tool ``ccusage`` parses the
    same per-message JSONL files that the ``claude`` binary writes under
    ``~/.claude`` and aggregates them into the same numbers the Max
    dashboard shows.  We shell out to it.

What this module returns:
    A dict suitable for merging into the gateway's ``_get_usage`` payload:

        {
            "max_week_cost": float USD spent this week,
            "max_today_cost": float USD spent today,
            "max_block_cost": float USD spent in the active 5h block,
            "max_block_remaining_minutes": int minutes until block reset,
            "max_total_tokens_week": int aggregate tokens this week,
        }

    Returns ``{}`` (not ``None``) on any failure so callers can ``.update()``
    without branching.

Caching:
    ccusage walks every JSONL file in HERMES_HOME-adjacent paths.  On a
    full session history that's slow (hundreds of ms to seconds).  We TTL
    cache for 30s — far below the 5-minute block reset granularity but
    cheap enough to update the status bar at human-perceptible cadence.

Failure modes (all return {}):
    - ccusage / npx not installed
    - npx download in progress and times out
    - Output schema changes (caller is defensive)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 60s amortises the ~12s npx ccusage cold-start across many status-bar
# repaints; the user only sees a refresh that often.  Drop to ~5s if the
# user installs ccusage globally (npx round-trip dominates today).
_CACHE_TTL_SEC = float(os.environ.get("HERMES_MAX_USAGE_CACHE_TTL", "60"))

# Hard cap so a hung ccusage process never blocks the status-bar refresh.
# Cold ``npx --yes ccusage@latest`` fetches the package and runs it — on
# this machine that takes ~12s.  A globally-installed ``ccusage`` binary
# returns in <2s.  The cap is set so npx works on the first call.
_INVOCATION_TIMEOUT_SEC = float(os.environ.get("HERMES_MAX_USAGE_TIMEOUT", "30"))

# Disable entirely for users who don't want the spawn-overhead. Falsy values
# are honoured — only "1"/"true"/"yes" enable, default ON.
_DISABLED = os.environ.get("HERMES_MAX_USAGE_DISABLED", "").strip() in ("1", "true", "yes")

_lock = threading.Lock()
_cache: Dict[str, Any] = {"value": {}, "expires_at": 0.0, "warned": False}


def _resolve_command() -> Optional[list[str]]:
    """Return the argv prefix for invoking ccusage, or None if unavailable.

    Prefers a globally-installed ``ccusage`` binary on PATH (faster: no
    npx round-trip).  Falls back to ``npx --yes ccusage@latest`` so the
    feature works without a separate install step.  Returns None if
    neither npx nor a plain node are available — the caller treats this
    as "unsupported" and skips the field.
    """
    if shutil.which("ccusage"):
        return ["ccusage"]
    if shutil.which("npx"):
        return ["npx", "--yes", "ccusage@latest"]
    return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _current_iso_week_key() -> str:
    """Return the ccusage week key (YYYY-MM-DD of the week's Sunday).

    ccusage groups weeks Sunday-to-Saturday in its default output.  We
    derive the Sunday of the current week by subtracting (weekday + 1) %
    7 days.  Python's ``datetime.weekday()`` returns Mon=0…Sun=6 — adjust
    so Sunday is week start.
    """
    now = _now_utc()
    # weekday(): Mon=0 ... Sun=6.  We want Sun=0 ... Sat=6.
    days_since_sunday = (now.weekday() + 1) % 7
    sunday = now.date().fromordinal(now.toordinal() - days_since_sunday)
    return sunday.isoformat()


def _parse_weekly(payload: Dict[str, Any]) -> Dict[str, float]:
    """Pick out the entry for the current week from a ``ccusage weekly --json`` payload.

    Schema (relevant fields):

        {"weekly":[{"week":"YYYY-MM-DD","totalCost":float,"totalTokens":int,...}]}

    Returns {} if the current week isn't in the payload yet (first call
    of a fresh week before any usage logged).
    """
    weeks = payload.get("weekly")
    if not isinstance(weeks, list) or not weeks:
        return {}
    target = _current_iso_week_key()
    for entry in weeks:
        if not isinstance(entry, dict):
            continue
        if entry.get("week") == target:
            return {
                "max_week_cost": float(entry.get("totalCost", 0) or 0),
                "max_total_tokens_week": int(entry.get("totalTokens", 0) or 0),
            }
    # Fallback: most recent week (last entry, since ccusage sorts oldest→newest).
    last = weeks[-1]
    if isinstance(last, dict):
        return {
            "max_week_cost": float(last.get("totalCost", 0) or 0),
            "max_total_tokens_week": int(last.get("totalTokens", 0) or 0),
        }
    return {}


def _parse_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pick out today's cost + the active 5h block from ``ccusage blocks --json --active``.

    Schema:
        {"blocks":[{"isActive":bool,"costUSD":float,"endTime":"ISO8601",...}]}
    """
    out: Dict[str, Any] = {}
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        return out

    now = _now_utc()
    for entry in blocks:
        if not isinstance(entry, dict):
            continue
        if not entry.get("isActive"):
            continue
        out["max_block_cost"] = float(entry.get("costUSD", 0) or 0)
        end_iso = entry.get("endTime")
        if isinstance(end_iso, str):
            try:
                # Accept both Z-suffix and +00:00 forms.
                end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                remaining = max(0, int((end_dt - now).total_seconds() // 60))
                out["max_block_remaining_minutes"] = remaining
            except ValueError:
                pass
        break  # only one active block at a time
    return out


def _run_ccusage(subcommand: str) -> Dict[str, Any]:
    """Run ``ccusage <subcommand> --json`` and return parsed payload, or {} on any error."""
    cmd = _resolve_command()
    if cmd is None:
        return {}
    argv = list(cmd) + [subcommand, "--json"]
    if subcommand == "blocks":
        argv.append("--active")
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_INVOCATION_TIMEOUT_SEC,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("claude_max_usage: ccusage %s failed: %s", subcommand, exc)
        return {}
    if proc.returncode != 0:
        logger.debug(
            "claude_max_usage: ccusage %s exit=%d stderr=%s",
            subcommand, proc.returncode, (proc.stderr or "")[:200],
        )
        return {}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        logger.debug("claude_max_usage: ccusage %s emitted non-JSON output", subcommand)
        return {}


def get_max_usage(force: bool = False) -> Dict[str, Any]:
    """Return cached Max-plan usage telemetry, refreshing if stale.

    Args:
        force: Skip TTL cache and re-query ccusage now.  Useful for the
               ``hermes max-usage`` CLI command.

    Returns:
        Dict with keys ``max_week_cost``, ``max_total_tokens_week``,
        ``max_block_cost``, ``max_block_remaining_minutes``,
        ``max_today_cost``.  Empty dict on failure / disabled.
    """
    if _DISABLED:
        return {}
    now = time.time()
    with _lock:
        if not force and _cache["expires_at"] > now and _cache["value"]:
            return dict(_cache["value"])

    weekly = _run_ccusage("weekly")
    blocks = _run_ccusage("blocks")

    out: Dict[str, Any] = {}
    out.update(_parse_weekly(weekly))
    out.update(_parse_blocks(blocks))

    # ``daily`` gives today's cost — same parse shape as weekly, indexed
    # on YYYY-MM-DD instead of week-Sunday.  We skip the spawn unless we
    # already know ccusage works (weekly succeeded) to avoid double the
    # spawn cost when the tool is missing.
    if weekly:
        daily = _run_ccusage("daily")
        days = daily.get("daily")
        if isinstance(days, list) and days:
            today_key = _now_utc().date().isoformat()
            for entry in days:
                if isinstance(entry, dict) and entry.get("date") == today_key:
                    out["max_today_cost"] = float(entry.get("totalCost", 0) or 0)
                    break

    with _lock:
        _cache["value"] = out
        _cache["expires_at"] = now + _CACHE_TTL_SEC
        if not out and not _cache["warned"]:
            logger.info(
                "claude_max_usage: no usage telemetry available "
                "(install ccusage globally with `npm i -g ccusage` for faster status-bar refresh, "
                "or set HERMES_MAX_USAGE_DISABLED=1 to suppress)"
            )
            _cache["warned"] = True
    return dict(out)


def format_status_bar_label(usage: Dict[str, Any]) -> str:
    """Render Max-usage fields as a single status-bar fragment.

    Format: ``Max $W/wk · $B blk · Hh Mm`` (only fields present).
    Returns ``""`` when no usage data available.
    """
    parts = []
    if "max_week_cost" in usage:
        parts.append(f"${usage['max_week_cost']:.2f}/wk")
    if "max_block_cost" in usage:
        parts.append(f"${usage['max_block_cost']:.2f} blk")
    if "max_block_remaining_minutes" in usage:
        m = usage["max_block_remaining_minutes"]
        if m >= 60:
            parts.append(f"{m // 60}h{m % 60:02d}m left")
        else:
            parts.append(f"{m}m left")
    if not parts:
        return ""
    return "Max " + " · ".join(parts)
