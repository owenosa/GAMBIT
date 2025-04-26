# RL1-ES.py
# Futures Directional ES Trader Agent Definition (Refactored)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ------------------
# Utility: Compute Counterfactuals
# ------------------
def compute_counterfactuals(action, next_price, entry_price, position_size, transaction_cost):
    cf = {}
    for d in [0,1,2]:
        if d != action['direction']:
            pos = 0 if d==0 else (1 if d==1 else -1)
            rtn = pos * ((next_price - entry_price) / entry_price)
            cf[f'dir_{d}'] = position_size * rtn - transaction_cost
    base_sign = 1 if action['direction']==1 else -1
    base_rtn = base_sign * ((next_price - entry_price) / entry_price)
    for frac in [0.5, 1.5]:
        alloc = position_size * frac
        cf[f'size_{frac}'] = alloc * base_rtn - transaction_cost
    orig_pct = action.get('stop_pct', 0.05)
    for pct in [orig_pct * 0.5, min(0.5, orig_pct * 2)]:
        sp = entry_price * (1 - pct) if action['direction']==1 else entry_price * (1 + pct)
        rtn = base_sign * ((sp - entry_price) / entry_price)
        cf[f'stop_{pct:.3f}'] = position_size * rtn - transaction_cost
    return cf

# ------------------
# TradingAgent Model
# ------------------
class TradingAgent(nn.Module):
    def __init__(self, input_size: int, shared_dim: int = 64, lstm_dim: int = 64):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, shared_dim),
            nn.ReLU()
        )
        self.macro_lstm = nn.LSTM(shared_dim, lstm_dim, batch_first=True)
        self.vol_lstm = nn.LSTM(shared_dim, lstm_dim, batch_first=True)
        self.price_lstm = nn.LSTM(shared_dim, lstm_dim, batch_first=True)
        self.direction_head = nn.Linear(lstm_dim, 2)
        self.size_head = nn.Linear(lstm_dim, 1)
        self.stop_head = nn.Linear(lstm_dim, 1)

    def forward(self, x):
        shared = self.shared_layer(x)
        macro_out, _ = self.macro_lstm(shared)
        vol_out, _ = self.vol_lstm(shared)
        price_out, _ = self.price_lstm(shared)
        macro_last = macro_out[:, -1, :]
        vol_last = vol_out[:, -1, :]
        price_last = price_out[:, -1, :]
        direction_logits = self.direction_head(macro_last)
        size_output = torch.sigmoid(self.size_head(vol_last))
        stop_output = F.relu(self.stop_head(price_last))
        return direction_logits, size_output, stop_output

    def sample_action(self, direction_logits, size_output, stop_output):
        dir_dist = Categorical(logits=direction_logits)
        direction = dir_dist.sample()
        logp = dir_dist.log_prob(direction)
        size_dist = Normal(size_output, 0.1)
        size = size_dist.sample().clamp(0, 1)
        logp += size_dist.log_prob(size).sum(-1)
        stop_dist = Normal(stop_output, 0.05)
        stop = stop_dist.sample().clamp(0, 0.5)
        logp += stop_dist.log_prob(stop).sum(-1)
        return {'direction': direction.item(), 'size': size.item(), 'stop_pct': stop.item()}, logp

def init_trading_agent(input_size: int, shared_dim: int = 64, lstm_dim: int = 64) -> TradingAgent:
    return TradingAgent(input_size, shared_dim, lstm_dim)

# ------------------
# Training Utilities
# ------------------

# Label Generation

def generate_labels(df):
    df['5d_forward_return'] = df['Price'].shift(-5) / df['Price'] - 1
    df['direction_label'] = (df['5d_forward_return'] > 0).astype(int)
    df['size_label'] = 1 / (df['Vol'] + 1e-6)
    df['size_label'] = df['size_label'].clip(0, 1)
    df['stop_label'] = abs(df['Price'] - df['Base']) / df['Price']
    df['stop_label'] = df['stop_label'].clip(0.01, 0.5)
    return df.dropna()


# Custom Dataset
class RLPretrainDataset(Dataset):
    def __init__(self, df, feature_cols, label_type):
        self.X = df[feature_cols].values
        self.label_type = label_type
        if label_type == 'direction':
            self.y = df['direction_label'].values
        elif label_type == 'size':
            self.y = df['size_label'].values
        elif label_type == 'stop':
            self.y = df['stop_label'].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        
        if self.label_type == 'direction':
            y = torch.tensor(self.y[idx], dtype=torch.long)  # <--- direction needs Long for CrossEntropy
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32)  # <--- size and stop stay Float

        return x, y


# Pretrain Routines

def pretrain_macro(agent, dataloader, epochs=5, lr=1e-3):
    for param in list(agent.vol_lstm.parameters()) + list(agent.size_head.parameters()) + \
                  list(agent.price_lstm.parameters()) + list(agent.stop_head.parameters()):
        param.requires_grad = False
    for param in list(agent.macro_lstm.parameters()) + list(agent.direction_head.parameters()) + \
                  list(agent.shared_layer.parameters()):
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        agent.train()
        total_loss = 0.0
        for x, y_direction in dataloader:
            logits, _, _ = agent(x)
            loss = loss_fn(logits, y_direction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Macro Pretrain Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def pretrain_vol(agent, dataloader, epochs=5, lr=1e-3):
    for param in list(agent.macro_lstm.parameters()) + list(agent.direction_head.parameters()) + \
                  list(agent.price_lstm.parameters()) + list(agent.stop_head.parameters()):
        param.requires_grad = False
    for param in list(agent.vol_lstm.parameters()) + list(agent.size_head.parameters()) + \
                  list(agent.shared_layer.parameters()):
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        agent.train()
        total_loss = 0.0
        for x, y_size in dataloader:
            _, size_output, _ = agent(x)
            loss = loss_fn(size_output.squeeze(), y_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Vol Pretrain Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def pretrain_price(agent, dataloader, epochs=5, lr=1e-3):
    for param in list(agent.macro_lstm.parameters()) + list(agent.direction_head.parameters()) + \
                  list(agent.vol_lstm.parameters()) + list(agent.size_head.parameters()):
        param.requires_grad = False
    for param in list(agent.price_lstm.parameters()) + list(agent.stop_head.parameters()) + \
                  list(agent.shared_layer.parameters()):
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        agent.train()
        total_loss = 0.0
        for x, y_stop in dataloader:
            _, _, stop_output = agent(x)
            loss = loss_fn(stop_output.squeeze(), y_stop)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Price Pretrain Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Unfreeze All Parameters for RL fine-tuning
def unfreeze_all(agent):
    for param in agent.parameters():
        param.requires_grad = True
