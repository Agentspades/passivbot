import argparse
import json
import os
import sys
import time
from typing import Any

from config_utils import load_config
from ai_agent import AIAgent


def _print_status(agent: AIAgent) -> None:
    print("AI Agent Status")
    print(f"  mode: {'shadow' if agent.shadow_mode else ('dry-run' if agent.cfg.get('dry_run') else 'live')}")
    print(f"  filters.vol_ema: {agent.filters.get('vol_ema')}")
    print(f"  filters.upnl_ema: {agent.filters.get('upnl_ema')}")
    print(f"  budgets.actions_this_hour: {agent.actions_this_hour}/{agent.budgets.get('max_actions_per_hour')}")
    print(f"  budgets.adjusts_this_hour: {agent.adjusts_this_hour}/{agent.budgets.get('max_adjusts_per_hour')}")
    print(f"  cooldown_until_ts: {agent.cooldown_until_ts} (now={time.time():.0f})")
    print(f"  paused_due_to_dd: {agent.paused_due_to_dd}")
    print(f"  paused_due_to_daily: {agent.paused_due_to_daily}")
    print(f"  blocking_due_to_vol: {agent.blocking_due_to_vol}")


def _tail_audit(path: str, n: int) -> None:
    try:
        if not os.path.exists(path):
            print(f"audit log not found: {path}")
            return
        with open(path, "r") as f:
            lines = f.readlines()[-n:]
        for ln in lines:
            try:
                obj = json.loads(ln)
                fields = {
                    k: obj.get(k)
                    for k in ["ts", "symbol", "unrealized_pnl_pct", "volatility", "decision"]
                }
                print(json.dumps(fields))
            except Exception:
                print(ln.rstrip())
    except Exception as e:
        print(f"error reading audit log: {e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI Agent CLI helpers")
    parser.add_argument("--config", default="configs/template.json", help="Path to config file")

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("print_status")
    p_tail = sub.add_parser("tail_audit")
    p_tail.add_argument("N", type=int, nargs="?", default=20, help="Number of lines")

    args = parser.parse_args(argv)
    config = load_config(args.config, live_only=True, verbose=False)
    agent = AIAgent(config)

    if args.cmd == "print_status":
        _print_status(agent)
    elif args.cmd == "tail_audit":
        audit_path = agent.cfg.get("audit_log_path", "logs/ai_agent_audit.jsonl")
        _tail_audit(audit_path, int(getattr(args, "N", 20)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

