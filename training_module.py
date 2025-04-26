# training_module.py
# Full pretraining + RL fine-tuning utilities

import importlib
import pandas as pd
import torch
import torch.optim as optim
from RL1 import load_rl_data, TradingEnv  # import env definitions

# -----------------
# Discount Rewards
# -----------------
def discount_rewards(rewards, gamma=0.99):
    """
    Compute discounted, normalized returns for a trajectory.
    """
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

# -----------------
# Run RL Fine-tuning
# -----------------
def run_training(env,
                 agent,
                 num_episodes: int = 100,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 max_seq: int = 120) -> pd.DataFrame:
    """
    RL fine-tuning loop: environment <-> agent interaction with policy-gradient updates.
    Logs per-episode performance and trades.
    """
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    performance_log = []

    for episode in range(num_episodes):
        obs = env.reset()
        obs_seq = []
        hidden = None
        done = False
        logps = []
        rewards = []
        initial_trades = env.trades

        while not done:
            obs_seq.append(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            if len(obs_seq) > max_seq:
                obs_seq = obs_seq[-max_seq:]
            x = torch.cat(obs_seq, dim=0).unsqueeze(0)

            logits, size_output, stop_output = agent(x)
            action, logp = agent.sample_action(logits, size_output, stop_output)

            obs, reward, done, info = env.step(action)

            logps.append(logp)
            rewards.append(reward)

        returns = discount_rewards(rewards, gamma)
        policy_loss = torch.stack([-lp * G for lp, G in zip(logps, returns)]).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        trades_executed = env.trades - initial_trades

        perf = info.get('performance', {})
        perf['episode'] = episode
        perf['trades'] = trades_executed
        performance_log.append(perf)

    return pd.DataFrame(performance_log)

# -----------------
# Full One-Click Setup (optional, still here if needed)
# -----------------
def RL1(ticker: str,
        rl_folder: str = 'RL Data',
        agent_folder: str = 'Agents',
        episodes: int = 100,
        max_seq: int = 120,
        gamma: float = 0.99,
        lr: float = 1e-3) -> pd.DataFrame:
    """
    Full entry point: loads environment, builds agent dynamically, runs RL training.
    """
    df = load_rl_data(rl_folder, ticker)
    env = TradingEnv(df)

    spec = f"{agent_folder}.r1_{ticker.lower()}"
    mod = importlib.import_module(spec)

    obs_dim = len(env.feature_cols) + 2  # features + Price + position_size
    agent = mod.init_trading_agent(input_size=obs_dim)

    return run_training(env,
                        agent,
                        num_episodes=episodes,
                        max_seq=max_seq,
                        gamma=gamma,
                        lr=lr)
