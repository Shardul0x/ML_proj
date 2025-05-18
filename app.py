import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from galaxy import GalaxyVAE, GalaxyPINN, simulate_evolution

# Setup
st.set_page_config(page_title="Galaxy Evolution Simulator", layout="wide")
st.title("🌌 Galaxy Morphology Evolution using VAE + PINN")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
    feature_cols = [
        'P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
        'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED'
    ]
    return df[feature_cols]

df = load_data()
st.sidebar.markdown("### Select a Galaxy Sample")
sample_idx = st.sidebar.slider("Sample Index", 0, len(df)-1, 0)
selected_feature = st.sidebar.selectbox("Feature to Visualize", df.columns)
steps = st.sidebar.slider("Evolution Steps", 5, 50, 20)

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Load Trained Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = GalaxyVAE().to(device)
try:
    vae.load_state_dict(torch.load("vae_galaxy.pth", map_location=device))
    vae.eval()
    st.success("VAE model loaded successfully")
except Exception as e:
    st.error(f"Failed to load VAE model: {e}")

pinn = GalaxyPINN().to(device)
try:
    pinn.load_state_dict(torch.load("pinn_galaxy.pth", map_location=device))
    pinn.eval()
    st.success("PINN model loaded successfully")
except Exception as e:
    st.error(f"Failed to load PINN model: {e}")

# ===== Visualizations =====

# Encode to Latent Space
with torch.no_grad():
    mu, _ = vae.encode(X_tensor.to(device))
    latent_np = mu.cpu().numpy()

st.subheader("Latent Space Representations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 2D Latent Space (PCA)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latent_np)
    fig1, ax1 = plt.subplots()
    ax1.scatter(reduced[:, 0], reduced[:, 1], c='indigo', alpha=0.6)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA Projection")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 3D Latent Space")
    from mpl_toolkits.mplot3d import Axes3D
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(latent_np[:, 0], latent_np[:, 1], latent_np[:, 2], c='darkcyan', alpha=0.6)
    ax2.set_title("3D Latent Space")
    st.pyplot(fig2)

# Reconstructed vs Original
st.subheader("Original vs Reconstructed Features")

with torch.no_grad():
    x_hat, _, _ = vae(X_tensor.to(device))
    original = X_tensor[sample_idx].cpu().numpy()
    reconstructed = x_hat[sample_idx].cpu().numpy()

fig3, ax3 = plt.subplots()
ax3.plot(original, label="Original", marker='o')
ax3.plot(reconstructed, label="Reconstructed", marker='x')
ax3.set_title(f"Sample {sample_idx} Comparison")
ax3.legend()
st.pyplot(fig3)

# Evolution Simulation
st.subheader("📈 Galaxy Feature Evolution Simulation")

sample = X_tensor[sample_idx].unsqueeze(0).to(device)
evolved_states, t_vals = simulate_evolution(pinn, sample, steps=steps)
evolved_states = np.array(evolved_states)[:, 0, :]

fig4, ax4 = plt.subplots()
ax4.plot(t_vals, evolved_states[:, df.columns.get_loc(selected_feature)], marker='o', color='teal')
ax4.set_xlabel("Time")
ax4.set_ylabel(selected_feature)
ax4.set_title(f"Evolution of {selected_feature} over Time")
ax4.grid(True)
st.pyplot(fig4)

# Optional: Display Feature Values Table
st.markdown("### Final Evolved Features")
final_state = pd.DataFrame([evolved_states[-1]], columns=df.columns)
st.dataframe(final_state.style.highlight_max(axis=1, color='lightgreen'))
