import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_audit(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    rows.sort(key=lambda x: x.get("ts", 0))
    return rows


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vs = sorted(values)
    idx = q * (len(vs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(vs) - 1)
    w = idx - lo
    return vs[lo] * (1 - w) + vs[hi] * w


def summarize(audit: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"pause": 0, "block": 0, "force_close": 0, "adjust": 0}
    pause_reasons: Dict[str, int] = defaultdict(int)
    vol_emas: List[float] = []
    upnl_emas: List[float] = []
    by_hour_actions: Dict[int, int] = defaultdict(int)
    by_hour_adjusts: Dict[int, int] = defaultdict(int)
    budget_skips = {"action": 0, "adjust": 0}

    for row in audit:
        ts = int(row.get("ts", 0))
        hour = (ts // 3600) * 3600
        dec = row.get("decision", {}) or {}
        if not isinstance(dec, dict):
            continue
        if dec.get("pause_bot"):
            counts["pause"] += 1
            pause_reasons[str(dec.get("pause_bot"))] += 1
            by_hour_actions[hour] += 1
        if dec.get("block_entry"):
            counts["block"] += 1
            by_hour_actions[hour] += 1
        if dec.get("force_close"):
            counts["force_close"] += 1
            by_hour_actions[hour] += 1
        if isinstance(dec.get("adjust"), dict):
            counts["adjust"] += 1
            by_hour_actions[hour] += 1
            by_hour_adjusts[hour] += 1
        if dec.get("info") == "budget_action_limit":
            budget_skips["action"] += 1
        if dec.get("info") == "budget_adjust_limit":
            budget_skips["adjust"] += 1
        try:
            filt = row.get("filters", {}) or {}
            if filt.get("vol_ema") is not None:
                vol_emas.append(float(filt.get("vol_ema")))
            if filt.get("upnl_ema") is not None:
                upnl_emas.append(float(filt.get("upnl_ema")))
        except Exception:
            pass

    hourly_counts = sorted(by_hour_actions.values()) if by_hour_actions else [0]
    hourly_adjust_counts = (
        sorted(by_hour_adjusts.values()) if by_hour_adjusts else [0]
    )

    summary: Dict[str, Any] = {
        "counts": counts,
        "pause_reasons": dict(pause_reasons),
        "filters": {
            "vol_ema_min": min(vol_emas) if vol_emas else None,
            "vol_ema_max": max(vol_emas) if vol_emas else None,
            "vol_ema_p50": percentile(vol_emas, 0.5) if vol_emas else None,
            "vol_ema_p90": percentile(vol_emas, 0.9) if vol_emas else None,
            "upnl_ema_min": min(upnl_emas) if upnl_emas else None,
            "upnl_ema_max": max(upnl_emas) if upnl_emas else None,
            "upnl_ema_p10": percentile(upnl_emas, 0.1) if upnl_emas else None,
        },
        "budgets": {
            "hourly_actions_p95": percentile(hourly_counts, 0.95) if hourly_counts else 0,
            "hourly_adjusts_p95": percentile(hourly_adjust_counts, 0.95)
            if hourly_adjust_counts
            else 0,
            "budget_skips": budget_skips,
        },
    }

    # Suggest calibrations conservatively
    suggestions: Dict[str, Any] = {"hysteresis": {}, "budgets": {}}
    if vol_emas:
        enter = float(summary["filters"]["vol_ema_p90"]) if summary["filters"]["vol_ema_p90"] is not None else None
        if enter:
            exit_ = round(0.75 * enter, 6)
            suggestions["hysteresis"]["volatility_enter"] = round(enter, 6)
            suggestions["hysteresis"]["volatility_exit"] = exit_
    # daily drawdown suggestions are data-dependent; leave None here
    # Budgets: 20% headroom above observed 95th percentile
    p95_actions = summary["budgets"]["hourly_actions_p95"] or 0
    p95_adjusts = summary["budgets"]["hourly_adjusts_p95"] or 0
    suggestions["budgets"]["max_actions_per_hour"] = int((p95_actions or 0) * 1.2 + 1)
    suggestions["budgets"]["max_adjusts_per_hour"] = int((p95_adjusts or 0) * 1.2 + 1)

    return {"summary": summary, "suggestions": suggestions}


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate AI Agent from audit log")
    ap.add_argument(
        "--audit", default="logs/ai_agent_audit.jsonl", help="Path to ai_agent_audit.jsonl"
    )
    ap.add_argument(
        "--output",
        default="logs/ai_agent_calibration.json",
        help="Where to write calibration summary",
    )
    ap.add_argument(
        "--last-hours",
        type=float,
        default=0.0,
        dest="last_hours",
        help="Only use the last N hours from the audit (default: all)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.audit):
        raise SystemExit(f"Audit log not found: {args.audit}")

    audit = load_audit(args.audit)
    if not audit:
        raise SystemExit("Audit log is empty; need 24h of data to calibrate")

    # Optionally filter by last-hours
    if args.last_hours and args.last_hours > 0 and audit:
        try:
            cutoff = time.time() - float(args.last_hours) * 3600.0
            audit = [row for row in audit if float(row.get("ts", 0)) >= cutoff]
        except Exception:
            pass
        if not audit:
            raise SystemExit("Audit window empty after --last-hours filter")

    result = summarize(audit)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
