import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os


def merging(land, ocean):
    """ merging the land and ocean csv file into one csv file
    * the result csv file contains difference between land and ocean temperature"""

    landTemp = land[["DATE", "TMAX"]].rename(columns={"TMAX": "TMAX_LAND"})
    oceanTemp = ocean[["DATE", "TMAX"]].rename(columns={"TMAX": "TMAX_OCEAN"})
    merged = pd.merge(
        landTemp,
        oceanTemp,
        on="DATE",
        how="inner"
    )

    merged["DIFF"] = merged["TMAX_OCEAN"] - merged["TMAX_LAND"]
    return merged

def graphing(df, name):
    """shows the graph of input csv file"""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    plt.figure(figsize=(10, 4))
    plt.plot(df["DATE"], df["DIFF"])
    plt.axhline(0)
    plt.xlabel("Date")
    plt.ylabel("Diff")
    plt.title(f"{name} Temperature Difference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)
        

def showMLP(df, weightPath):
    """showing graph of MLP trended value
    * if pth file is in weightPath -> load the weight
    * else -> learning the MLP and save pth file"""
    
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")

    X = df["DATE"].map(pd.Timestamp.toordinal).to_numpy()
    y = df["DIFF"].to_numpy(dtype=float)

    window = 15
    y_smooth = pd.Series(y).rolling(window=window, center=True).mean()
    mask = ~y_smooth.isna()
    X_fit = X[mask]
    y_fit = y_smooth[mask].to_numpy()
        
    X_t = torch.tensor(X_fit, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)

    X_mean, X_std = X_t.mean(), X_t.std()
    y_mean, y_std = y_t.mean(), y_t.std()

    eps = 1e-8
    if X_std < eps:
        X_std = torch.tensor(eps)
    if y_std < eps:
        y_std = torch.tensor(eps)

    X_n = (X_t - X_mean) / X_std
    y_n = (y_t - y_mean) / y_std

    model = MLP()

    if os.path.isfile(weightPath):
        checkpoint = torch.load(weightPath, map_location="cpu")
        window = checkpoint.get("window", window)
        model.load_state_dict(checkpoint["model_state_dict"])
        X_mean = checkpoint["X_mean"]
        X_std = checkpoint["X_std"]
        y_mean = checkpoint["y_mean"]
        y_std = checkpoint["y_std"]
        model.eval()

    else:
        lr = 0.01
        epochs = 3000
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_n)
            loss = loss_fn(pred, y_n)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss {loss.item():.4f}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "window": window
            },
            weightPath
        )

        model.eval()

    X_grid = np.linspace(X_fit.min(), X_fit.max(), 800)
    Xg_t = torch.tensor(X_grid, dtype=torch.float32).view(-1, 1)
    Xg_n = (Xg_t - X_mean) / X_std

    with torch.no_grad():
        yg = model(Xg_n) * y_std + y_mean

    dates_grid = [pd.Timestamp.fromordinal(int(v)) for v in X_grid]

    plt.figure(figsize=(10, 5))
    plt.scatter(df["DATE"], df["DIFF"], s=5, alpha=0.3, label="Observed")
    plt.plot(dates_grid, yg.numpy().ravel(), color="red", linewidth=2, label="MLP Trend")
    plt.axhline(0)
    plt.xlabel("Date")
    plt.ylabel("Temperature Difference")
    plt.legend()
    plt.tight_layout()

    plt.show()
