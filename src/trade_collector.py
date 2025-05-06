#!/usr/bin/env python3
import os
import json
import time
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from typing import Dict, List, Any


class TradeCollector:
    def __init__(self, api_keys_file: str, output_dir: str):
        self.api_keys_file = api_keys_file
        self.output_dir = output_dir
        self.exchanges = {}
        self.last_fetch_time = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    async def setup(self):
        """Load API keys and initialize exchange connections"""
        with open(self.api_keys_file, "r") as f:
            api_keys = json.load(f)

        for user, user_data in api_keys.items():
            if "bybit" in user_data:
                # Initialize Bybit exchange for each user
                self.exchanges[user] = ccxt.bybit(
                    {
                        "apiKey": user_data["bybit"]["key"],
                        "secret": user_data["bybit"]["secret"],
                        "enableRateLimit": True,
                        "options": {"defaultType": "swap"},  # For V5 API
                    }
                )
                self.last_fetch_time[user] = datetime.now() - timedelta(days=7)

    async def fetch_trades(self, user: str, since_time: datetime) -> List[Dict]:
        """Fetch trades for a specific user since the given time"""
        exchange = self.exchanges[user]

        # Convert datetime to milliseconds timestamp
        since_ms = int(since_time.timestamp() * 1000)

        all_trades = []
        try:
            # Fetch closed positions (V5 API)
            closed_positions = await exchange.privateGetV5PositionClosedPnl(
                {"category": "linear", "startTime": since_ms}
            )

            if (
                closed_positions
                and "result" in closed_positions
                and "list" in closed_positions["result"]
            ):
                all_trades.extend(closed_positions["result"]["list"])

            # Also fetch recent trades
            trades = await exchange.privateGetV5OrderHistory(
                {"category": "linear", "startTime": since_ms}
            )

            if trades and "result" in trades and "list" in trades["result"]:
                all_trades.extend(trades["result"]["list"])

        except Exception as e:
            print(f"Error fetching trades for {user}: {e}")

        return all_trades

    def process_trades(self, trades: List[Dict]) -> pd.DataFrame:
        """Process raw trade data into a structured DataFrame"""
        if not trades:
            return pd.DataFrame()

        # Extract relevant fields from trade data
        processed_trades = []
        for trade in trades:
            try:
                processed_trade = {
                    "symbol": trade.get("symbol"),
                    "side": trade.get("side"),
                    "price": float(trade.get("price", 0)),
                    "qty": float(trade.get("qty", 0)),
                    "realized_pnl": float(trade.get("closedPnl", 0)),
                    "fee": float(trade.get("fee", 0)),
                    "timestamp": int(trade.get("createdTime", 0)),
                    "order_type": trade.get("orderType"),
                    "position_idx": trade.get("positionIdx"),
                }
                processed_trades.append(processed_trade)
            except (ValueError, TypeError) as e:
                print(f"Error processing trade: {e}")

        return pd.DataFrame(processed_trades)

    async def collect_and_save(self):
        """Collect trades from all accounts and save to CSV files"""
        for user, exchange in self.exchanges.items():
            try:
                since_time = self.last_fetch_time[user]
                print(f"Fetching trades for {user} since {since_time}")

                trades = await self.fetch_trades(user, since_time)
                df = self.process_trades(trades)

                if not df.empty:
                    # Save to CSV with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.output_dir}/{user}_trades_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved {len(df)} trades to {filename}")

                    # Update last fetch time
                    self.last_fetch_time[user] = datetime.now()
                else:
                    print(f"No new trades found for {user}")

            except Exception as e:
                print(f"Error collecting trades for {user}: {e}")

    async def run_periodic(self, interval_hours=6):
        """Run collection periodically at specified interval"""
        await self.setup()

        while True:
            await self.collect_and_save()
            print(f"Sleeping for {interval_hours} hours...")
            await asyncio.sleep(interval_hours * 3600)

    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            await exchange.close()


async def main():
    collector = TradeCollector(
        api_keys_file="api-keys.json", output_dir="data/collected_trades"
    )

    try:
        await collector.run_periodic(interval_hours=6)
    except KeyboardInterrupt:
        print("Collection stopped by user")
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
