import os
import numpy as np
import pandas as pd
import random
import yfinance as yf

def load_rl_data(folder_path: str, ticker: str) -> pd.DataFrame:
    """
    Load RL feature CSV and merge daily OHLC price data. Expects columns:
    Date, Macro, Vol, Security, Security Proba, Price, AVWAP, Base, R1, R2, S1, S2
    """
    path = os.path.join(folder_path, f"RL - {ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    # keep regime signals and pivots as features
    feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Price','Volume']]
    merged = df[feature_cols + ['Price']].dropna()
    return merged

class TradingEnv:
    """
    Backtesting environment with phased reward logic.
    Phases:
      0: step-wise P/L (clip ±2)
      1: end-of-trade P/L (clip ±3) + transaction penalty (-1 each trade)
      2: Sortino-adjusted reward at episode end (clip ±4)
      3: Regime-alignment bonus + drawdown penalties
    Manual phase toggle via `env.phase = N`.
    """
    def __init__(self, df: pd.DataFrame, episode_length: int = 378,
                 transaction_cost: float = 15.0, seed: int | None = None):
        self.full_df = df.reset_index(drop=True)
        self.feature_cols = [c for c in df.columns if c != 'Price']
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.initial_capital = 100_000.0
        self.seed(seed)

        # Phase control & bonus weights
        self.phase = 0
        self.b1 = 1.0  # directional bonus
        self.b2 = 2.0  # sizing bonus
        self.b3 = 1.0  # stop-loss bonus
        # Track equity peak for drawdown penalty
        self.peak_equity = self.initial_capital

    def seed(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return seed

    def reset(self) -> np.ndarray:
        idx = random.randint(0, len(self.full_df) - self.episode_length)
        self.episode = self.full_df.iloc[idx:idx + self.episode_length].reset_index(drop=True)

        # Initialize state variables
        self.step_i = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.equity_curve = [self.equity]
        self.position = 0
        self.position_size = 0.0
        self.entry_price = None
        self.stop_price = None
        self.trades = 0
        self.rewards = []
        self.done = False

        return self._obs()

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, dict]:
        d = int(action['direction'])    # 0=hold,1=long,2=short
        sz = float(action['size'])
        sp = float(action['stop_pct'])
        row = self.episode.iloc[self.step_i]
        price = float(row['Price'])

        base_reward = 0.0
        traded = False

        # 1) Stop-loss exit
        if self.position:
            if (self.position == 1 and price <= self.stop_price) or \
               (self.position == -1 and price >= self.stop_price):
                rtn = self.position * ((self.stop_price - self.entry_price) / self.entry_price)
                pnl = self.position_size * rtn
                base_reward += pnl - self.transaction_cost
                self._reset_position()
                traded = True

        # 2) Allow immediate re-entry after stop loss exit
        # (no additional changes needed - because after _reset_position, self.position == 0)

        # 3) Entry
        if self.position == 0 and d in (1, 2):
            alloc = self.equity * sz
            base_reward -= self.transaction_cost
            self.position = 1 if d == 1 else -1
            self.position_size = alloc
            self.entry_price = price

            if self.position == 1:
                self.stop_price = price * (1 - abs(sp))
                if self.stop_price >= price:
                    self.stop_price = price * 0.99
            elif self.position == -1:
                self.stop_price = price * (1 + abs(sp))
                if self.stop_price <= price:
                    self.stop_price = price * 1.01

            self.trades += 1
            traded = True

        # 4) Mark-to-market (existing position)
        elif self.position and self.entry_price is not None:
            rtn = self.position * ((price - self.entry_price) / self.entry_price)
            base_reward += self.position_size * rtn

        # 5) End-of-episode forced exit
        if self.step_i == self.episode_length - 1 and self.position:
            rtn = self.position * ((price - self.entry_price) / self.entry_price)
            pnl = self.position_size * rtn
            base_reward += pnl - self.transaction_cost
            self._reset_position()
            self.done = True
            traded = True

        # 6) Phase-based reward
        reward = self._apply_phase_reward(base_reward, action, traded)

        # NEW: Penalty for doing "nothing" (holding flat) for too long
        if self.position == 0 and d == 0:
            reward -= 0.05  # small negative reward for pure inactivity (only if not in a position)

        # 7) Equity update & drawdown tracking
        self.equity += reward
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

        # 8) Drawdown penalty
        if self.phase == 3 and self.equity < 0.9 * self.peak_equity:
            reward -= 5.0
        if self.equity <= 0:
            reward -= 10.0
            self.done = True

        # 9) Record reward
        self.rewards.append(reward)

        # 10) Advance step index if not done
        if not self.done:
            self.step_i += 1

        # Prepare info
        info = {'equity': self.equity}
        if self.done:
            info['performance'] = self._compute_performance_metrics()

        return self._obs(), reward, self.done, info

    def _apply_phase_reward(self, base_reward: float, action: dict, traded: bool) -> float:
        # Phase 0: step-wise P/L + small trade bonus
        if self.phase == 0:
            r = float(np.clip(base_reward, -2.0, 5.0))
            if traded:
                r += 0.5  # bonus for taking trades
            return r

        # Phase 1: end-of-trade P/L + tx penalty
        if self.phase == 1:
            r = float(np.clip(base_reward if traded else 0.0, -3.0, 3.0))
            if traded:
                r -= 1.0
            return r
        # Phase 2: include Sortino at end
        if self.phase == 2:
            if not self.done:
                return self._apply_phase_reward(base_reward, action, traded)
            rets = np.diff(self.equity_curve) / self.equity_curve[:-1]
            down = rets[rets < 0]
            downside = np.std(down) if len(down) else 1e-6
            sortino = (np.mean(rets) / downside) * np.sqrt(252)
            return float(np.clip(sortino, -4.0, 4.0))
        # Phase 3: regime bonuses + phase 2 logic
        r = self._apply_phase_reward(base_reward, action, traded if not self.done else traded)
        if traded:
            macro = self.episode.iloc[self.step_i]['Macro']
            if (macro > 0 and action['direction'] == 1) or (macro <= 0 and action['direction'] == 2):
                r += self.b1
            vol = self.episode.iloc[self.step_i]['Vol']
            if (vol > 0.5 and action['size'] < 0.5) or (vol <= 0.5 and action['size'] >= 0.5):
                r += self.b2
            sec = self.episode.iloc[self.step_i]['Security']
            if (sec < 0.5 and action['stop_pct'] < 0.1) or (sec >= 0.5 and action['stop_pct'] >= 0.1):
                r += self.b3
        return r

    def _obs(self) -> np.ndarray:
        row = self.episode.iloc[self.step_i]
        vals = [row[c] for c in self.feature_cols]
        return np.array(vals + [row['Price'], self.position_size], dtype=np.float32)

    def _reset_position(self):
        self.position = 0
        self.position_size = 0.0
        self.entry_price = None
        self.stop_price = None

    def _compute_performance_metrics(self) -> dict:
        eq = np.array(self.equity_curve)
        rets = eq[1:] / eq[:-1] - 1
        return {
            'total_return': eq[-1] / eq[0] - 1,
            'sharpe':       rets.mean() / (rets.std() + 1e-6) * np.sqrt(252),
            'max_drawdown': (eq - np.maximum.accumulate(eq)).min(),
        }
