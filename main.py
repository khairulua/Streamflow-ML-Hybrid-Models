"""
Created on 2025-05-28

by: Md Khairul Amin
Coastal Hydrology Lab,
The University of Alabama
"""

import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def pbias(y_true, y_pred):
    return 100 * np.sum(y_pred - y_true) / np.sum(y_true)

def create_sequences(data, input_len=30, output_len=1):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.dropout(x)
        return self.transformer(x)

class AnomalyAwareLSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, anomaly_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.anomaly_score = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.memory_proj = nn.Linear(input_dim, anomaly_dim)
        self.decoder = nn.Linear(hidden_dim + anomaly_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        scores = self.anomaly_score(x).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        memory = torch.sum(weights * self.memory_proj(x), dim=1)
        last_hidden = lstm_out[:, -1, :]
        return self.decoder(torch.cat([last_hidden, memory], dim=-1))

class StreamflowTransformerLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = AnomalyAwareLSTM()

    def forward(self, x):
        return self.decoder(self.encoder(x))

def run_pipeline(data_path, input_len=30, batch_size=32, patience=10, epochs=30):
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.interpolate(method='linear').dropna()

    all_results_test = []
    all_results_train = []
    prediction_records_by_gauge = {}

    for gauge in df.columns:
        print(f"\n\U0001F9E0 Training models for gauge: {gauge}")
        series = df[gauge].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(series).flatten()
        X, y = create_sequences(scaled, input_len)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        gauge_predictions = []

        model_defs = {
            'LSTM': lambda: Sequential([LSTM(64, input_shape=(input_len, 1)), Dense(1)]),
            'BiLSTM': lambda: Sequential([Bidirectional(LSTM(64), input_shape=(input_len, 1)), Dense(1)]),
            'Stacked LSTM': lambda: Sequential([
                LSTM(64, return_sequences=True, input_shape=(input_len, 1)),
                LSTM(32),
                Dense(1)
            ]),
            'X-LSTM': lambda: Sequential([
                Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_len, 1)),
                LSTM(64),
                Dense(1)
            ])
        }

        for name, build_model in model_defs.items():
            model = build_model()
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0)

            y_pred_test = model.predict(X_test).flatten()
            y_pred_train = model.predict(X_train).flatten()

            y_pred_test_inv = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
            y_true_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_train_inv = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
            y_true_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

            all_results_test.append({"Gauge": gauge, "Model": name, "RMSE": round(np.sqrt(mean_squared_error(y_true_test_inv, y_pred_test_inv)), 3), "NSE": round(nse(y_true_test_inv, y_pred_test_inv), 3), "PBIAS": round(pbias(y_true_test_inv, y_pred_test_inv), 3)})
            all_results_train.append({"Gauge": gauge, "Model": name, "RMSE": round(np.sqrt(mean_squared_error(y_true_train_inv, y_pred_train_inv)), 3), "NSE": round(nse(y_true_train_inv, y_pred_train_inv), 3), "PBIAS": round(pbias(y_true_train_inv, y_pred_train_inv), 3)})

            for t, p in zip(y_true_test_inv, y_pred_test_inv):
                gauge_predictions.append({"Model": name, "Target": t, "Prediction": p})

        # PyTorch model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = StreamflowTransformerLSTM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(), yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            y_pred_test = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
            y_pred_train = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        y_pred_test_inv = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        y_true_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_train_inv = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
        y_true_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

        all_results_test.append({"Gauge": gauge, "Model": "Transformer+CustomLSTM", "RMSE": round(np.sqrt(mean_squared_error(y_true_test_inv, y_pred_test_inv)), 3), "NSE": round(nse(y_true_test_inv, y_pred_test_inv), 3), "PBIAS": round(pbias(y_true_test_inv, y_pred_test_inv), 3)})
        all_results_train.append({"Gauge": gauge, "Model": "Transformer+CustomLSTM", "RMSE": round(np.sqrt(mean_squared_error(y_true_train_inv, y_pred_train_inv)), 3), "NSE": round(nse(y_true_train_inv, y_pred_train_inv), 3), "PBIAS": round(pbias(y_true_train_inv, y_pred_train_inv), 3)})

        for t, p in zip(y_true_test_inv, y_pred_test_inv):
            gauge_predictions.append({"Model": "Transformer+CustomLSTM", "Target": t, "Prediction": p})

        prediction_records_by_gauge[gauge] = pd.DataFrame(gauge_predictions)

    pd.DataFrame(all_results_test).to_csv("results_test.csv", index=False)
    pd.DataFrame(all_results_train).to_csv("results_train.csv", index=False)

    with pd.ExcelWriter("predictions_by_gauge.xlsx") as writer:
        for gauge, df_sheet in prediction_records_by_gauge.items():
            sheet_name = gauge[:31].replace("/", "-").replace("\\", "-")
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n\u2705 All model results saved: results_test.csv, results_train.csv, predictions_by_gauge.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample_combined_daily.csv")
    args = parser.parse_args()
    run_pipeline(args.data_path)
