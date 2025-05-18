import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, accuracy_score
from PIL import Image
import glob
import random
import seaborn as sns

# ====== Setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ====== Data Loading ======
df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
df = df[feature_cols].dropna()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
X_noisy = X_tensor + 0.01 * torch.randn_like(X_tensor)

# ====== Model Definitions ======
class GalaxyVAE(nn.Module):
    def __init__(self, input_dim=9, latent_dim=3):
        super(GalaxyVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class GalaxyPINN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64):
        super(GalaxyPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x_t = torch.cat([x, t], dim=1)
        return self.net(x_t)


def physics_loss(preds, t, weight=0.1):
    d_preds_dt = torch.autograd.grad(outputs=preds, inputs=t, grad_outputs=torch.ones_like(preds), create_graph=True)[0]
    return weight * torch.mean(d_preds_dt**2)

# ====== Training ======
vae = GalaxyVAE().to(device)
vae_optim = optim.Adam(vae.parameters(), lr=0.001)

pinn = GalaxyPINN().to(device)
pinn_optim = optim.Adam(pinn.parameters(), lr=0.001)

X_noisy.requires_grad = True
timesteps = torch.linspace(0, 1, X_tensor.size(0)).unsqueeze(1).to(device).requires_grad_()

# Loss functions
def vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return 0.9 * recon + 0.1 * kl

# Training VAE
print("🔄 Training VAE...")
vae.train()
for epoch in range(200):
    vae_optim.zero_grad()
    x_hat, mu, logvar = vae(X_noisy)
    loss = vae_loss(X_noisy, x_hat, mu, logvar)
    loss.backward()
    vae_optim.step()
    if (epoch + 1) % 20 == 0:
        print(f"VAE Epoch {epoch+1}, Loss: {loss.item():.2f}")
torch.save(vae.state_dict(), "models/vae_galaxy.pth")
print("✅ VAE model saved!")

# Training PINN
print("🔄 Training PINN...")
pinn.train()
for epoch in range(400):
    pinn_optim.zero_grad()
    out = pinn(X_noisy, timesteps)
    loss = nn.MSELoss()(out, X_noisy) + physics_loss(out, timesteps)
    loss.backward()
    pinn_optim.step()
    if (epoch + 1) % 20 == 0:
        print(f"PINN Epoch {epoch+1}, Loss: {loss.item():.4f}")
torch.save(pinn.state_dict(), "models/pinn_galaxy.pth")
print("✅ PINN model saved!")

# ====== Evaluation ======
def evaluate_vae(vae, data, threshold=0.5):
    vae.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = vae(data)
        mse = nn.MSELoss()(reconstructed, data).item()
        rmse = np.sqrt(mse)
        actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
        pred_binary = (reconstructed.detach().cpu().numpy() > threshold).astype(int)
        accuracy = accuracy_score(actual_binary.flatten(), pred_binary.flatten())
        f1 = f1_score(actual_binary.flatten(), pred_binary.flatten(), average='micro')
        cm = confusion_matrix(actual_binary.flatten(), pred_binary.flatten())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('VAE Confusion Matrix')
        plt.savefig('results/vae_confusion_matrix.png')
        plt.close()
        return {'MSE': mse, 'RMSE': rmse, 'Accuracy': accuracy, 'F1 Score': f1}

def evaluate_pinn(pinn, data, t_values=None, threshold=0.5):
    pinn.eval()
    if t_values is None:
        t_values = torch.linspace(0, 1, 10).to(device)
    results = []
    with torch.no_grad():
        for t in t_values:
            t_batch = t.expand(data.size(0), 1)
            preds = pinn(data, t_batch)
            mse = nn.MSELoss()(preds, data).item()
            rmse = np.sqrt(mse)
            actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
            pred_binary = (preds.detach().cpu().numpy() > threshold).astype(int)
            accuracy = accuracy_score(actual_binary.flatten(), pred_binary.flatten())
            f1 = f1_score(actual_binary.flatten(), pred_binary.flatten(), average='micro')
            results.append({'t': t.item(), 'MSE': mse, 'RMSE': rmse, 'Accuracy': accuracy, 'F1 Score': f1})
        return results

# ====== Run Evaluation ======
print("\n📊 Evaluating VAE...")
vae_metrics = evaluate_vae(vae, X_tensor)
print(f"VAE MSE: {vae_metrics['MSE']:.4f}, RMSE: {vae_metrics['RMSE']:.4f}, Accuracy: {vae_metrics['Accuracy']:.4f}, F1 Score: {vae_metrics['F1 Score']:.4f}")

print("\n📊 Evaluating PINN...")
pinn_metrics = evaluate_pinn(pinn, X_tensor)
for r in pinn_metrics:
    print(f"t={r['t']:.2f} | MSE={r['MSE']:.4f}, RMSE={r['RMSE']:.4f}, Accuracy={r['Accuracy']:.4f}, F1={r['F1 Score']:.4f}")


# ====== Plotting PINN Metrics ======
def plot_pinn_metrics(pinn_metrics):
    t_values = [r['t'] for r in pinn_metrics]
    metrics = ['MSE', 'RMSE', 'Accuracy', 'F1 Score']

    plt.figure(figsize=(12, 10))
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        values = [r[metric] for r in pinn_metrics]
        plt.plot(t_values, values, 'o-', label=metric)
        plt.xlabel('Time (t)')
        plt.ylabel(metric)
        plt.title(f'{metric} over Time')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('results/pinn_metrics_over_time.png')
    plt.close()
    print("✅ PINN metrics over time plot saved at results/pinn_metrics_over_time.png")

# ====== Run Evaluation ======
print("\n📊 Evaluating VAE...")
vae_metrics = evaluate_vae(vae, X_tensor)
print(f"VAE MSE: {vae_metrics['MSE']:.4f}, RMSE: {vae_metrics['RMSE']:.4f}, Accuracy: {vae_metrics['Accuracy']:.4f}, F1 Score: {vae_metrics['F1 Score']:.4f}")

print("\n📊 Evaluating PINN...")
pinn_metrics = evaluate_pinn(pinn, X_tensor)
for r in pinn_metrics:
    print(f"t={r['t']:.2f} | MSE={r['MSE']:.4f}, RMSE={r['RMSE']:.4f}, Accuracy={r['Accuracy']:.4f}, F1={r['F1 Score']:.4f}")

# ====== Plot PINN Metrics ======
plot_pinn_metrics(pinn_metrics)




