# FIXED GPT ADVISOR SERVICE (DAILY ONLY)
# Only runs at 00:00 (UTC+8), no more realtime GPT calls

import os
import sqlite3
import datetime
from openai import OpenAI

DB_PATH = os.getenv("DB_PATH", "/data/trading_data.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TIMEZONE_OFFSET = 8  # Taiwan UTC+8

client = OpenAI(api_key=OPENAI_API_KEY)


class GPTAdvisorService:

    def __init__(self):
        self.last_run_date = None

    def should_run_daily(self):
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=TIMEZONE_OFFSET)
        today = now.date()

        if now.hour == 0 and now.minute < 5:
            if self.last_run_date != today:
                self.last_run_date = today
                return True

        return False

    def fetch_today_trades(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        today = (datetime.datetime.utcnow() + datetime.timedelta(hours=TIMEZONE_OFFSET)).date()

        cursor.execute("""
            SELECT symbol, pnl, leverage, margin, fee, entry_time, exit_time
            FROM trades
            WHERE DATE(entry_time) = ?
        """, (today,))

        rows = cursor.fetchall()
        conn.close()
        return rows

    def summarize(self, trades):
        if not trades:
            return {"msg": "No trades today"}

        total_pnl = sum([t[1] for t in trades])
        total_fee = sum([t[4] for t in trades])
        win = len([t for t in trades if t[1] > 0])
        loss = len([t for t in trades if t[1] <= 0])

        return {
            "total_trades": len(trades),
            "win": win,
            "loss": loss,
            "pnl": total_pnl,
            "fee": total_fee,
        }

    def call_gpt(self, summary):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": "你是交易AI優化顧問"},
                    {"role": "user", "content": f"今日交易數據：{summary}，請給優化建議"}
                ]
            )
            return response.output[0].content[0].text

        except Exception as e:
            print("GPT ERROR:", e)
            return None

    def save_advice(self, advice):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gpt_advice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("INSERT INTO gpt_advice (content) VALUES (?)", (advice,))
        conn.commit()
        conn.close()

    def run(self):
        if not self.should_run_daily():
            return

        print("🧠 GPT DAILY REVIEW START")

        trades = self.fetch_today_trades()
        summary = self.summarize(trades)

        advice = self.call_gpt(summary)

        if advice:
            self.save_advice(advice)
            print("✅ GPT advice saved")

        print("🧠 GPT DAILY REVIEW END")
