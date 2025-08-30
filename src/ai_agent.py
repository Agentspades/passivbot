import logging
import json
import os
import time
from datetime import datetime, timezone, time as dtime
from typing import Any, Dict
from collections import defaultdict
from pure_funcs import symbol_to_coin


def _ema(prev: float | None, value: float | None, alpha: float) -> float | None:
    try:
        if value is None:
            return prev
        if prev is None:
            return float(value)
        return float(alpha) * float(value) + (1.0 - float(alpha)) * float(prev)
    except Exception:
        return prev


class AIAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        live = config.get("live", {})
        self.enabled = bool(live.get("enable_ai_agent", True)) and bool(
            live.get("ai_agent", {}).get("enabled", True)
        )
        ai_cfg = live.get("ai_agent", {})
        # base config
        self.cfg = {
            "dry_run": bool(live.get("ai_agent", {}).get("dry_run", False)),
            "max_drawdown_pct": float(
                ai_cfg.get("max_drawdown_pct", 0.12)
            ),
            "max_daily_loss_pct": float(
                ai_cfg.get("max_daily_loss_pct", 0.06)
            ),
            "max_trade_loss_pct": float(
                ai_cfg.get("max_trade_loss_pct", 0.03)
            ),
            "min_interval_secs": float(
                ai_cfg.get("min_interval_secs", 5.0)
            ),
            "cooldown_secs_after_force_close": float(
                ai_cfg.get("cooldown_secs_after_force_close", 120.0)
            ),
            "cooldown_secs_after_adjust": float(
                ai_cfg.get("cooldown_secs_after_adjust", 60.0)
            ),
            "param_caps": ai_cfg.get("param_caps", {}),
            "volatility_threshold": float(
                ai_cfg.get("volatility_threshold", 0.08)
            ),
            "losing_streak_window": int(
                ai_cfg.get("losing_streak_window", 5)
            ),
            "losing_streak_adjustment": float(
                ai_cfg.get("losing_streak_adjustment", 0.2)
            ),
            "audit_log_path": ai_cfg.get(
                "audit_log_path", "logs/ai_agent_audit.jsonl"
            ),
            "kill_switch": bool(ai_cfg.get("kill_switch", False)),
        }
        # New knobs: hysteresis, smoothing, budgets, shadow mode, symbol overrides
        hyst = ai_cfg.get("hysteresis", {}) or {}
        smooth = ai_cfg.get("smoothing", {}) or {}
        budgets = ai_cfg.get("budgets", {}) or {}
        self.hysteresis = {
            "volatility_enter": float(hyst.get("volatility_enter", 0.08)),
            "volatility_exit": float(
                hyst.get(
                    "volatility_exit",
                    max(0.0, 0.75 * float(hyst.get("volatility_enter", 0.08))),
                )
            ),
            "dd_pause_enter": float(hyst.get("dd_pause_enter", 0.12)),
            "dd_pause_exit": float(
                hyst.get(
                    "dd_pause_exit",
                    max(0.0, 0.8 * float(hyst.get("dd_pause_enter", 0.12))),
                )
            ),
        }
        self.smoothing = {
            "vol_ema_alpha": float(smooth.get("vol_ema_alpha", 0.2)),
            "pnl_ema_alpha": float(smooth.get("pnl_ema_alpha", 0.15)),
        }
        self.budgets = {
            "max_actions_per_hour": int(budgets.get("max_actions_per_hour", 12)),
            "max_adjusts_per_hour": int(budgets.get("max_adjusts_per_hour", 8)),
        }
        self.shadow_mode = bool(ai_cfg.get("shadow_mode", False))
        self.symbol_overrides = ai_cfg.get("symbol_overrides", {}) or {}
        # Optional gates/halts
        self.halts = ai_cfg.get("halts", []) or []
        self.gates = ai_cfg.get("gates", {}) or {}

        # Filters and gating states
        self.filters: Dict[str, Any] = {"vol_ema": None, "upnl_ema": None}
        self.blocking_due_to_vol: bool = False
        self.paused_due_to_dd: bool = False
        self.paused_due_to_daily: bool = False
        self.last_dd_ratio: float | None = None
        self._last_log_block_state: bool | None = None
        self._last_log_pause_state: bool | None = None

        # Action budgets state
        self.hour_window_start_ts = int(time.time() // 3600) * 3600
        self.actions_this_hour = 0
        self.adjusts_this_hour = 0

        # Per-symbol tracking
        self.per_symbol = defaultdict(lambda: {"losing_streak": 0, "last_adjust_ts": 0})

        # Mode label for logs
        self._mode_label = (
            "shadow" if self.shadow_mode else ("dry-run" if self.cfg.get("dry_run") else "live")
        )
        self.thresholds = {}
        if self.enabled:
            logging.info("AI Agent initialized and observing trades...")
            logging.info(
                f"AI Agent safety guardrails active: max_dd={self.cfg['max_drawdown_pct']}, max_daily_loss={self.cfg['max_daily_loss_pct']}, max_trade_loss={self.cfg['max_trade_loss_pct']}, vol_enter={self.hysteresis['volatility_enter']} vol_exit={self.hysteresis['volatility_exit']}"
            )

        # internal state
        self.last_action_ts = 0.0
        self.cooldown_until_ts = 0.0
        self.session_peak_equity = 0.0
        self.daily_pnl_tracker = 0.0
        self.last_utc_date = None
        self.losing_streak = 0
        self.audit_log_path = self.cfg.get("audit_log_path")

    def observe(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Update EMAs if present
        try:
            vol_raw = state.get("volatility")
            self.filters["vol_ema"] = _ema(
                self.filters.get("vol_ema"),
                None if vol_raw is None else float(vol_raw),
                self.smoothing["vol_ema_alpha"],
            )
        except Exception:
            pass
        try:
            upnl_raw = float(state.get("account", {}).get("unrealized_pnl_pct", 0.0) or 0.0)
            self.filters["upnl_ema"] = _ema(
                self.filters.get("upnl_ema"), upnl_raw, self.smoothing["pnl_ema_alpha"],
            )
        except Exception:
            pass
        return state

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision: Dict[str, Any] = {}
        now = time.time()

        if not self.enabled:
            return decision

        # Pre-check gate
        try:
            if self.cfg.get("kill_switch"):
                return {"pause_bot": "kill_switch"}
        except Exception:
            pass

        # Soft gates/halts: skip entries in configured windows or bad microstructure
        try:
            # Event halts
            if self._is_in_halt(now):
                decision["block_entry"] = True
                try:
                    logging.info("AI Agent: event halt active; blocking entries")
                except Exception:
                    pass
            # Spread gate if present in state and configured
            max_spread = self.gates.get("max_spread_pct") if isinstance(self.gates, dict) else None
            if max_spread is not None:
                try:
                    spread = state.get("spread_pct")
                    if spread is None and isinstance(state.get("market"), dict):
                        spread = state["market"].get("spread_pct")
                    if spread is not None and float(spread) > float(max_spread):
                        decision["block_entry"] = True
                        logging.info(
                            f"AI Agent: spread gate block: spread_pct={float(spread):.6f} max={float(max_spread):.6f}"
                        )
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.cooldown_until_ts and now < self.cooldown_until_ts:
                return {"info": "cooldown"}
        except Exception:
            pass

        try:
            if now - self.last_action_ts < float(
                self.cfg.get("min_interval_secs", 5.0)
            ):
                return {}
        except Exception:
            pass

        # Risk Guardrail – Force Close on large drawdown of unrealized PnL (per-position also below)
        try:
            upnl_pct = float(
                state.get("account", {}).get("unrealized_pnl_pct", 0.0) or 0.0
            )
            if upnl_pct < -float(self.cfg.get("max_trade_loss_pct", 0.03)):
                decision["force_close"] = True
        except Exception:
            pass

        # Risk Guardrail – Block Entries on high volatility using EMA + hysteresis
        try:
            vol_ema = self.filters.get("vol_ema")
            if vol_ema is not None:
                if not self.blocking_due_to_vol and vol_ema >= self.hysteresis["volatility_enter"]:
                    self.blocking_due_to_vol = True
                    try:
                        logging.info(
                            f"AI Agent: mode={self._mode_label} block entries: vol_ema={vol_ema:.3f} enter={self.hysteresis['volatility_enter']} exit={self.hysteresis['volatility_exit']}"
                        )
                    except Exception:
                        pass
                elif self.blocking_due_to_vol and vol_ema <= self.hysteresis["volatility_exit"]:
                    self.blocking_due_to_vol = False
                    try:
                        logging.info(
                            f"AI Agent: mode={self._mode_label} resume entries: vol_ema={vol_ema:.3f} enter={self.hysteresis['volatility_enter']} exit={self.hysteresis['volatility_exit']}"
                        )
                    except Exception:
                        pass
            # fallback to legacy threshold if we still don't have vol ema
            elif state.get("volatility") is not None:
                volatility = float(state.get("volatility"))
                if volatility > float(self.cfg.get("volatility_threshold", 0.08)):
                    self.blocking_due_to_vol = True
            # if currently blocking, emit action to block entries this tick
            if self.blocking_due_to_vol:
                decision["block_entry"] = True
        except Exception:
            pass

        # Global loss guardrails
        try:
            balance = float(state.get("account", {}).get("balance", 0.0) or 0.0)
            equity = balance
            try:
                upnl_pct_acc = float(
                    state.get("account", {}).get("unrealized_pnl_pct", 0.0) or 0.0
                )
                equity = balance * (1.0 + upnl_pct_acc)
            except Exception:
                pass
            if equity > self.session_peak_equity:
                self.session_peak_equity = equity
            if self.session_peak_equity > 0:
                dd = equity / self.session_peak_equity - 1.0
                dd_ratio = max(0.0, -float(dd))  # positive drawdown ratio
                self.last_dd_ratio = dd_ratio
                # Hysteresis-based pause/resume on session DD
                if not self.paused_due_to_dd and dd_ratio >= self.hysteresis["dd_pause_enter"]:
                    self.paused_due_to_dd = True
                    try:
                        logging.info(
                            f"AI Agent: mode={self._mode_label} pause due to DD: dd={dd_ratio:.3f} enter={self.hysteresis['dd_pause_enter']} exit={self.hysteresis['dd_pause_exit']}"
                        )
                    except Exception:
                        pass
                    decision["pause_bot"] = "session_dd"
                elif self.paused_due_to_dd and dd_ratio <= self.hysteresis["dd_pause_exit"]:
                    # Only clear flag here; actual resume happens in maybe_resume when cooldown elapsed
                    self.paused_due_to_dd = False
                    try:
                        logging.info(
                            f"AI Agent: mode={self._mode_label} DD below exit: dd={dd_ratio:.3f} enter={self.hysteresis['dd_pause_enter']} exit={self.hysteresis['dd_pause_exit']}"
                        )
                    except Exception:
                        pass
                # Legacy immediate pause if exceeding absolute max dd
                if dd_ratio >= float(self.cfg.get("max_drawdown_pct", 0.12)):
                    decision["pause_bot"] = "max_drawdown"
        except Exception:
            pass

        try:
            # daily loss tracker reset at UTC midnight
            now_utc_date = datetime.now(timezone.utc).date()
            if self.last_utc_date is None or now_utc_date != self.last_utc_date:
                self.last_utc_date = now_utc_date
                self.daily_pnl_tracker = 0.0
                # clear daily pause at new UTC day
                self.paused_due_to_daily = False
            # Use realized-only daily PnL if available
            realized_today_pct = state.get("account", {}).get("realized_pnl_pct_today")
            try:
                realized_today_pct = (
                    float(realized_today_pct) if realized_today_pct is not None else None
                )
            except Exception:
                realized_today_pct = None
            loss_thr = float(self.cfg.get("max_daily_loss_pct", 0.06))
            if (
                realized_today_pct is not None
                and (not self.paused_due_to_daily)
                and realized_today_pct <= -loss_thr
            ):
                decision["pause_bot"] = "daily_loss"
                self.paused_due_to_daily = True
        except Exception:
            pass

        # per-trade hard stop
        try:
            for pos in state.get("open_positions", []) or []:
                up = float(pos.get("upnl_pct", 0.0) or 0.0)
                if up <= -float(self.cfg.get("max_trade_loss_pct", 0.03)):
                    decision["force_close"] = True
                    break
        except Exception:
            pass

        # Adaptive Grid Adjustment – losing streak per symbol in last N closed trades
        try:
            closed_trades = state.get("closed_trades", [])
            window = int(self.cfg.get("losing_streak_window", 5))
            # compute per-symbol losing streak across last 10 trades default for stronger signal
            by_symbol: Dict[str, list] = defaultdict(list)
            for ct in closed_trades[-10:]:
                try:
                    sym = ct.get("symbol") or "multi"
                    by_symbol[sym].append(ct)
                except Exception:
                    continue
            propose_adjust = False
            adjust: Dict[str, Any] = {}
            affected_symbol = None
            for sym, trades in by_symbol.items():
                lastn = trades[-window:] if len(trades) >= window else []
                if lastn and all([(ct.get("pnl_pct", 0.0) or 0.0) < 0.0 for ct in lastn]):
                    self.per_symbol[sym]["losing_streak"] = len(lastn)
                    propose_adjust = True
                    affected_symbol = sym
                    break
            if propose_adjust:
                live_cfg = state.get("live_config", {})
                curr_long = (
                    live_cfg.get("long", {}).get("grid_spacing_pct")
                    if isinstance(live_cfg, dict)
                    else None
                )
                curr_short = (
                    live_cfg.get("short", {}).get("grid_spacing_pct")
                    if isinstance(live_cfg, dict)
                    else None
                )
                if curr_long is None:
                    try:
                        curr_long = float(self.config["bot"]["long"]["entry_grid_spacing_pct"])  # type: ignore[index]
                    except Exception:
                        curr_long = None
                if curr_short is None:
                    try:
                        curr_short = float(self.config["bot"]["short"]["entry_grid_spacing_pct"])  # type: ignore[index]
                    except Exception:
                        curr_short = None

                def reduce_pct(x):
                    try:
                        reduction = float(self.cfg.get("losing_streak_adjustment", 0.2))
                        return max(0.1, float(x) * (1.0 - reduction))
                    except Exception:
                        return None

                new_long = reduce_pct(curr_long)
                new_short = reduce_pct(curr_short)
                # apply per-symbol overrides if present
                if affected_symbol and affected_symbol in self.symbol_overrides:
                    ov = self.symbol_overrides[affected_symbol] or {}
                    caps = (ov.get("param_caps") or {}).copy()
                    for key, nv in {"long.grid_spacing_pct": new_long, "short.grid_spacing_pct": new_short}.items():
                        if nv is None:
                            continue
                        side, param = key.split(".")
                        bounds = caps.get(side, {}).get(param, {}) if isinstance(caps, dict) else {}
                        try:
                            vmin = float(bounds.get("min", nv))
                            vmax = float(bounds.get("max", nv))
                            nv = max(vmin, min(vmax, float(nv)))
                        except Exception:
                            pass
                        if side == "long":
                            new_long = nv
                        else:
                            new_short = nv

                if new_long is not None:
                    adjust["long.grid_spacing_pct"] = new_long
                if new_short is not None:
                    adjust["short.grid_spacing_pct"] = new_short
                if adjust:
                    decision["adjust"] = adjust
                    if affected_symbol:
                        decision["_affected_symbol"] = affected_symbol
        except Exception:
            pass

        # Rescue Mode – per-symbol reduce-only acceleration when underwater in benign volatility
        try:
            adjust2: Dict[str, Any] = {}
            rescue: Dict[str, Any] = {"engage": [], "release": []}
            # Determine if volatility is benign enough
            vema = self.filters.get("vol_ema")
            vol_ok = True
            if vema is not None:
                try:
                    vol_ok = (
                        (not self.blocking_due_to_vol)
                        and vema <= self.hysteresis["volatility_exit"] * self.rescue_cfg["max_vol_mult"]
                    )
                except Exception:
                    vol_ok = not self.blocking_due_to_vol
            # Build map of open positions from state
            open_pos_map: Dict[tuple, Dict[str, float]] = {}
            for p in state.get("open_positions", []) or []:
                try:
                    coin = symbol_to_coin(p.get("symbol", ""))
                    side = str(p.get("position_side") or "")
                    upnl = float(p.get("upnl_pct", 0.0) or 0.0)
                    size = float(p.get("size", 0.0) or 0.0)
                except Exception:
                    continue
                if coin and side:
                    open_pos_map[(coin, side)] = {"upnl": upnl, "size": size}
            # Engage
            if vol_ok:
                for (coin, side), meta in open_pos_map.items():
                    if meta["size"] == 0.0:
                        continue
                    try:
                        if meta["upnl"] <= -float(self.rescue_cfg["trigger_loss_pct"]):
                            if (coin, side) not in self._rescue_state:
                                rescue["engage"].append({"coin": coin, "side": side})
                                tgt = float(self.rescue_cfg["targets"]["close_grid_qty_pct"])
                                adjust2[f"coin_overrides.{coin}.bot.{side}.close_grid_qty_pct"] = tgt
                    except Exception:
                        continue
            # Release
            for key in list(self._rescue_state.keys()):
                coin, side = key
                meta = open_pos_map.get((coin, side))
                if (meta is None) or (meta.get("size", 0.0) == 0.0) or (
                    float(meta.get("upnl", 0.0)) >= -float(self.rescue_cfg["release_loss_pct"])
                ):
                    base = (self._rescue_state.get(key, {}) or {}).get("baseline", {})
                    if "close_grid_qty_pct" in base:
                        adjust2[f"coin_overrides.{coin}.bot.{side}.close_grid_qty_pct"] = float(
                            base["close_grid_qty_pct"]
                        )
                    rescue["release"].append({"coin": coin, "side": side})
            if rescue["engage"] or rescue["release"]:
                decision["rescue"] = rescue
                if adjust2:
                    decision.setdefault("adjust", {}).update(adjust2)
        except Exception:
            pass

        # audit logging per decision
        try:
            if decision:
                os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
                payload = {
                    "ts": time.time(),
                    "symbol": decision.get("_affected_symbol") or "multi",
                    "equity": float(
                        state.get("account", {}).get("balance", 0.0) or 0.0
                    ),
                    "unrealized_pnl_pct": float(
                        state.get("account", {}).get("unrealized_pnl_pct", 0.0) or 0.0
                    ),
                    "volatility": state.get("volatility"),
                    "losing_streak": self.losing_streak,
                    "decision": decision,
                    "mode": self._mode_label,
                    "filters": {
                        "vol_ema": self.filters.get("vol_ema"),
                        "upnl_ema": self.filters.get("upnl_ema"),
                    },
                }
                with open(self.audit_log_path, "a") as f:
                    f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

        # Enforce action budgets before returning an actionable decision (skip in shadow mode)
        try:
            self._ensure_budget_window()
            if not self.shadow_mode:
                would_adjust = isinstance(decision.get("adjust"), dict) or bool(decision.get("rescue"))
                would_act = bool(
                    decision.get("pause_bot")
                    or decision.get("block_entry")
                    or decision.get("force_close")
                    or would_adjust
                )
                if would_act:
                    if self.actions_this_hour + 1 > self.budgets["max_actions_per_hour"]:
                        logging.info(
                            f"AI Agent: mode={self._mode_label} budget action-skip: actions_this_hour={self.actions_this_hour} max={self.budgets['max_actions_per_hour']}"
                        )
                        return {"info": "budget_action_limit"}
                    if would_adjust and (self.adjusts_this_hour + 1 > self.budgets["max_adjusts_per_hour"]):
                        logging.info(
                            f"AI Agent: mode={self._mode_label} budget adjust-skip: adjusts_this_hour={self.adjusts_this_hour} max={self.budgets['max_adjusts_per_hour']}"
                        )
                        # remove adjust but keep other actions if any
                        decision.pop("adjust", None)
                        decision["info"] = "budget_adjust_limit"
        except Exception:
            pass

        return decision

    def _is_in_halt(self, now_epoch: float) -> bool:
        try:
            if not self.halts:
                return False
            now_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
            # Absolute windows first
            for h in self.halts:
                if not isinstance(h, dict):
                    continue
                st = h.get("start_ts")
                et = h.get("end_ts")
                if st is not None and et is not None:
                    try:
                        if float(st) <= now_epoch < float(et):
                            return True
                    except Exception:
                        pass
            # Daily windows (UTC): {start_utc: "HH:MM", end_utc: "HH:MM", weekdays: [0..6]}
            for h in self.halts:
                if not isinstance(h, dict):
                    continue
                s = h.get("start_utc")
                e = h.get("end_utc")
                if not s or not e:
                    continue
                wd = h.get("weekdays")
                if isinstance(wd, list) and len(wd) > 0:
                    try:
                        if now_dt.weekday() not in set(int(x) for x in wd):
                            continue
                    except Exception:
                        pass
                try:
                    sh, sm = map(int, str(s).split(":", 1))
                    eh, em = map(int, str(e).split(":", 1))
                    start_t = dtime(sh, sm)
                    end_t = dtime(eh, em)
                    tnow = now_dt.time()
                    if start_t <= end_t:
                        if start_t <= tnow < end_t:
                            return True
                    else:
                        # window wraps midnight
                        if tnow >= start_t or tnow < end_t:
                            return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def apply(self, decision: Dict[str, Any], strategy: Any, trader: Any) -> None:
        acted = False

        try:
            logging.debug(f"AI Agent apply called with decision: {decision}")
        except Exception:
            pass

        try:
            # Pause Bot
            if isinstance(decision, dict) and decision.get("pause_bot"):
                reason = decision.get("pause_bot")
                try:
                    if self.shadow_mode:
                        logging.info("AI Agent: SHADOW MODE – action suppressed (pause)")
                    else:
                        setattr(trader, "paused", True)
                        setattr(trader, "skip_entry", True)
                        logging.info(
                            f"AI Agent: mode={self._mode_label} PAUSE due to {reason}"
                        )
                except Exception as e:
                    logging.error(f"AI Agent pause error: {e}")
                acted = True

            # Block Entry
            if isinstance(decision, dict) and decision.get("block_entry"):
                try:
                    if self.shadow_mode:
                        logging.info(
                            "AI Agent: SHADOW MODE – action suppressed (block entries)"
                        )
                    else:
                        setattr(trader, "skip_entry", True)
                        logging.info(
                            f"AI Agent: mode={self._mode_label} Blocking entries due to high volatility"
                        )
                except Exception as e:
                    logging.error(f"AI Agent block_entry error: {e}")
                acted = True

            # Force Close
            if isinstance(decision, dict) and decision.get("force_close"):
                try:
                    if self.shadow_mode:
                        logging.info(
                            "AI Agent: SHADOW MODE – action suppressed (force close)"
                        )
                    else:
                        if self.cfg.get("dry_run"):
                            logging.info(
                                "AI Agent: [DRY-RUN] Force closing positions due to risk rule"
                            )
                        else:
                            if hasattr(trader, "close_all_positions") and callable(trader.close_all_positions):
                                logging.info("AI Agent: Force closing positions due to risk rule")
                                _ = trader.close_all_positions()
                                # enter paused state to allow safe resume after cooldown
                                try:
                                    setattr(trader, "paused", True)
                                    setattr(trader, "skip_entry", True)
                                except Exception:
                                    pass
                            else:
                                # Fallback: engage per-symbol graceful_stop on symbols with open positions
                                try:
                                    positions = getattr(trader, "positions", {}) or {}
                                    applied = []
                                    for sym, pos in positions.items():
                                        try:
                                            if float(pos.get("long", {}).get("size", 0.0)) != 0.0:
                                                coin = symbol_to_coin(sym)
                                                self.safe_set_param(
                                                    None,
                                                    trader,
                                                    f"coin_overrides.{coin}.live.forced_mode_long",
                                                    "graceful_stop",
                                                )
                                                applied.append((coin, "long"))
                                        except Exception:
                                            pass
                                        try:
                                            if float(pos.get("short", {}).get("size", 0.0)) != 0.0:
                                                coin = symbol_to_coin(sym)
                                                self.safe_set_param(
                                                    None,
                                                    trader,
                                                    f"coin_overrides.{coin}.live.forced_mode_short",
                                                    "graceful_stop",
                                                )
                                                applied.append((coin, "short"))
                                        except Exception:
                                            pass
                                    if applied:
                                        logging.info(
                                            f"AI Agent: Per-symbol graceful_stop engaged: {sorted(applied)}"
                                        )
                                    else:
                                        # As a last resort, toggle global graceful_stop
                                        self.safe_set_param(None, trader, "live.forced_mode_long", "graceful_stop")
                                        self.safe_set_param(None, trader, "live.forced_mode_short", "graceful_stop")
                                        logging.info(
                                            "AI Agent: Force close fallback engaged (global graceful_stop)"
                                        )
                                except Exception as e:
                                    logging.info(
                                        f"AI Agent: Force close requested but encountered error: {e}"
                                    )
                    acted = True
                    if not self.shadow_mode:
                        self.cooldown_until_ts = time.time() + float(
                            self.cfg.get("cooldown_secs_after_force_close", 120.0)
                        )
                except Exception as e:
                    logging.error(f"AI Agent force_close error: {e}")

            # Adjust parameters
            if isinstance(decision, dict) and isinstance(decision.get("adjust"), dict):
                try:
                    if self.shadow_mode:
                        logging.info("AI Agent: SHADOW MODE – action suppressed (adjust)")
                    else:
                        for k, v in decision["adjust"].items():
                            self.safe_set_param(strategy, trader, k, v)

                    try:
                        lg = decision["adjust"].get("long.grid_spacing_pct")
                        sg = decision["adjust"].get("short.grid_spacing_pct")
                        if lg is not None and sg is not None and float(lg) == float(sg):
                            logging.info(
                                f"AI Agent: mode={self._mode_label} Adjusted grid spacing to {lg}"
                            )
                        else:
                            if lg is not None:
                                logging.info(
                                    f"AI Agent: mode={self._mode_label} Adjusted long grid spacing to {lg}"
                                )
                            if sg is not None:
                                logging.info(
                                    f"AI Agent: mode={self._mode_label} Adjusted short grid spacing to {sg}"
                                )
                    except Exception:
                        pass
                    acted = True
                    if not self.shadow_mode:
                        self.cooldown_until_ts = time.time() + float(
                            self.cfg.get("cooldown_secs_after_adjust", 60.0)
                        )
                except Exception as e:
                    logging.error(f"AI Agent adjust error: {e}")
        except Exception as e:
            logging.error(f"AI Agent apply outer error: {e}")

        if acted:
            self.last_action_ts = time.time()
            # increment budgets counters for realized actions (skip in shadow)
            if not self.shadow_mode:
                try:
                    self._ensure_budget_window()
                    self.actions_this_hour += 1
                    if isinstance(decision.get("adjust"), dict):
                        self.adjusts_this_hour += 1
                except Exception:
                    pass

            # Rescue metadata bookkeeping (store baselines on engage, cleanup on release)
            try:
                resc = decision.get("rescue") if isinstance(decision, dict) else None
                if isinstance(resc, dict):
                    # On engage, record baselines for keys we override
                    for item in resc.get("engage", []) or []:
                        try:
                            coin = item.get("coin")
                            side = item.get("side")
                            if not coin or not side:
                                continue
                            key = (coin, side)
                            if key not in self._rescue_state:
                                # Read current baseline of close_grid_qty_pct via config, preferring override if any
                                baseline = {}
                                try:
                                    sym = trader.coin_to_symbol(coin)
                                    # prefer override if present
                                    val = None
                                    try:
                                        val = trader.config.get("coin_overrides", {}).get(coin, {}).get("bot", {}).get(side, {}).get("close_grid_qty_pct")
                                    except Exception:
                                        val = None
                                    if val is None:
                                        val = trader.config_get(["bot", side, "close_grid_qty_pct"], symbol=sym)
                                    baseline["close_grid_qty_pct"] = float(val)
                                except Exception:
                                    pass
                                self._rescue_state[key] = {"baseline": baseline}
                        except Exception:
                            continue
                    # On release, clear state entries
                    for item in resc.get("release", []) or []:
                        try:
                            coin = item.get("coin")
                            side = item.get("side")
                            if not coin or not side:
                                continue
                            self._rescue_state.pop((coin, side), None)
                        except Exception:
                            continue
            except Exception:
                pass

    def maybe_resume(self, trader: Any) -> None:
        """Attempt safe resume if paused_due_to_dd cleared and cooldown elapsed.

        This is called each tick after evaluate/apply.
        """
        try:
            # Only resume when cooldown elapsed
            now = time.time()
            if now >= float(self.cooldown_until_ts or 0):
                # If paused_due_to_dd cleared and daily loss not in effect
                if not self.paused_due_to_dd and not self.paused_due_to_daily:
                    try:
                        # if globally paused due to a hard close_all_positions path, resume
                        if getattr(trader, "paused", False):
                            setattr(trader, "paused", False)
                            setattr(trader, "skip_entry", False)
                        # clear forced modes we may have set during fallback
                        try:
                            live_cfg = getattr(self, "config", {}).get("live", {})
                            for mode_name in ("panic", "graceful_stop"):
                                if live_cfg.get("forced_mode_long") == mode_name:
                                    self.safe_set_param(None, trader, "live.forced_mode_long", "")
                                if live_cfg.get("forced_mode_short") == mode_name:
                                    self.safe_set_param(None, trader, "live.forced_mode_short", "")
                        except Exception:
                            pass
                        # Clear per-symbol graceful_stop when their positions are closed
                        try:
                            cfg = getattr(trader, "config", {})
                            co = cfg.get("coin_overrides", {}) or {}
                            positions = getattr(trader, "positions", {}) or {}
                            open_long = {symbol_to_coin(s) for s, p in positions.items() if float(p.get("long", {}).get("size", 0.0)) != 0.0}
                            open_short = {symbol_to_coin(s) for s, p in positions.items() if float(p.get("short", {}).get("size", 0.0)) != 0.0}
                            changed = False
                            for coin, ov in list(co.items()):
                                if not isinstance(ov, dict):
                                    continue
                                live = ov.get("live", {})
                                if live.get("forced_mode_long") == "graceful_stop" and coin not in open_long:
                                    co.setdefault(coin, {}).setdefault("live", {})["forced_mode_long"] = ""
                                    changed = True
                                if live.get("forced_mode_short") == "graceful_stop" and coin not in open_short:
                                    co.setdefault(coin, {}).setdefault("live", {})["forced_mode_short"] = ""
                                    changed = True
                            if changed:
                                logging.info("AI Agent: cleared per-symbol graceful_stop where positions closed")
                        except Exception:
                            pass
                        logging.info(
                            f"AI Agent: mode={self._mode_label} resume/cleanup complete (cooldown elapsed)"
                        )
                        # reset cooldown
                        self.cooldown_until_ts = 0.0
                    except Exception:
                        pass
        except Exception:
            pass

    def safe_set_param(self, strategy: Any, trader: Any, path: str, value: Any) -> None:
        try:
            if self.cfg.get("dry_run"):
                logging.info(f"AI Agent: [DRY-RUN] Would set {path} -> {value}")
                return
            side = None
            param = path
            if "." in path:
                side, param = path.split(".", 1)
            capped_value = value
            try:
                caps = self.cfg.get("param_caps", {})
                if side in ("long", "short") and param in (
                    "grid_spacing_pct",
                    "pos_size_pct",
                ):
                    bounds = caps.get(side, {}).get(param, {})
                    vmin = float(bounds.get("min", value))
                    vmax = float(bounds.get("max", value))
                    capped_value = max(vmin, min(vmax, float(value)))
            except Exception:
                capped_value = value

            done = False
            if strategy is not None:
                try:
                    if side and hasattr(strategy, side):
                        obj = getattr(strategy, side)
                        if hasattr(obj, param):
                            old = getattr(obj, param)
                            setattr(obj, param, capped_value)
                            logging.info(
                                f"AI Agent: mode={self._mode_label} Adjusted {side}.{param} {old} -> {capped_value}"
                            )
                            done = True
                    else:
                        attr_name = path.replace(".", "_")
                        if hasattr(strategy, attr_name):
                            old = getattr(strategy, attr_name)
                            setattr(strategy, attr_name, capped_value)
                            logging.info(
                                f"AI Agent: mode={self._mode_label} Adjusted {attr_name} {old} -> {capped_value}"
                            )
                            done = True
                except Exception:
                    pass

            if not done and hasattr(trader, "config"):
                try:
                    if side in ("long", "short"):
                        if param == "grid_spacing_pct":
                            old = trader.config["bot"][side].get(
                                "entry_grid_spacing_pct"
                            )
                            trader.config["bot"][side]["entry_grid_spacing_pct"] = (
                                float(capped_value)
                            )
                            logging.info(
                                f"AI Agent: mode={self._mode_label} Adjusted config.bot.{side}.entry_grid_spacing_pct {old} -> {capped_value}"
                            )
                        elif param == "pos_size_pct":
                            old = trader.config["bot"][side].get(
                                "entry_initial_qty_pct"
                            )
                            trader.config["bot"][side]["entry_initial_qty_pct"] = float(
                                capped_value
                            )
                            logging.info(
                                f"AI Agent: mode={self._mode_label} Adjusted config.bot.{side}.entry_initial_qty_pct {old} -> {capped_value}"
                            )
                    elif side == "live":
                        old = trader.config["live"].get(param)
                        trader.config["live"][param] = capped_value
                        logging.info(
                            f"AI Agent: mode={self._mode_label} Adjusted config.live.{param} {old} -> {capped_value}"
                        )
                    elif side == "coin_overrides":
                        # Path format: coin_overrides.<coin>.<rest>
                        try:
                            parts = param.split(".")
                            if len(parts) >= 2:
                                coin = parts[0]
                                subpath = parts[1:]
                                cfg = getattr(trader, "config", {})
                                node = cfg.setdefault("coin_overrides", {}).setdefault(coin, {})
                                for p in subpath[:-1]:
                                    node = node.setdefault(p, {})
                                leaf = subpath[-1]
                                old = node.get(leaf)
                                node[leaf] = capped_value
                                # refresh trader.coin_overrides mapping if method exists
                                try:
                                    if hasattr(trader, "init_coin_overrides") and callable(trader.init_coin_overrides):
                                        trader.init_coin_overrides()
                                except Exception:
                                    pass
                                logging.info(
                                    f"AI Agent: mode={self._mode_label} Adjusted coin_overrides.{coin}.{'.'.join(subpath)} {old} -> {capped_value}"
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"AI Agent safe_set_param error: {e}")

    def _ensure_budget_window(self) -> None:
        try:
            now = int(time.time())
            hour_start = (now // 3600) * 3600
            if hour_start != self.hour_window_start_ts:
                self.hour_window_start_ts = hour_start
                self.actions_this_hour = 0
                self.adjusts_this_hour = 0
        except Exception:
            pass
