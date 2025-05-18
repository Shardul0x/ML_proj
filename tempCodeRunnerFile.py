import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from galaxy import GalaxyVAE, GalaxyPINN, simulate_evolution
import os

# Use all CPU threads (for any NumPy or sklearn parallelism)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True
torch.set_num_threads(os.cpu_count())

# === Load Dataset ===
print("🔄 Loading Galaxy metadata...")
df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
feature_cols = [
    'P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
    'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED'
]
df = df[feature_cols]

# === User Input ===
print("\n🪐 Enter metadata for prediction:")
meta_input = [float(input(f"{col}: ")) for col in feature_cols]

# === Scaling ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled[:1000], dtype=torch.float32)  # latent space
sample_scaled = scaler.transform([meta_input])
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(0)

# === Load Models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

vae = GalaxyVAE().to(device)
vae.load_state_dict(torch.load("C:/Users/ASUS/Downloads/ML/models/vae_galaxy.pth", map_location=device))
vae.eval()
print("✅ VAE model loaded")

pinn = GalaxyPINN().to(device)
pinn.load_state_dict(torch.load("C:/Users/ASUS/Downloads/ML/models/pinn_galaxy.pth", map_location=device))
pinn.eval()
print("✅ PINN model loaded")

# === Latent Vector (skip plotting for speed) ===
with torch.no_grad():
    mu, _ = vae.encode(X_tensor.to(device))
    mu = mu[~torch.isnan(mu).any(dim=1)]

# === Reconstruction (skip plotting for speed) ===
with torch.no_grad():
    recon, _, _ = vae(sample_tensor.to(device))

# === Galaxy Evolution ===
print("🔁 Simulating evolution...")
with torch.no_grad():
    evolved_states, t_vals = simulate_evolution(pinn, sample_tensor.to(device), steps=20)
evolved_states = np.array(evolved_states)[:, 0, :]

# === Final Output ===
final_df = pd.DataFrame([evolved_states[-1]], columns=feature_cols)
print("\n✅ Final evolved state:")
print(final_df.round(3).T)

print("\n🚀 Done!")
