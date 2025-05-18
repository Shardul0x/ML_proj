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


device = torch.device("cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
                'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
df = df[feature_cols].dropna()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)


class GalaxyVAE(nn.Module):
    def __init__(self, input_dim=9, latent_dim=3):
        super(GalaxyVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim), nn.Sigmoid()
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
    def __init__(self, input_dim=9, hidden_dim=32):
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
    d_preds_dt = torch.autograd.grad(
        outputs=preds, inputs=t,
        grad_outputs=torch.ones_like(preds),
        create_graph=True
    )[0]
    return weight * torch.mean(d_preds_dt**2)


def draw_spiral_galaxy_features(step, features):
    P_EL, P_CW, P_ACW, P_EDGE, P_DK, P_MG, P_CS, P_EL_DEBIASED, P_CS_DEBIASED = features
    num_arms = int(2 + 4 * (P_CW + P_ACW))  # 2–6 arms
    spiral_tightness = 0.5 + 3 * (1 - P_EL)  # tighter for spirals
    brightness = 0.5 + P_CS  # brighter center

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor('black')
    num_stars = 1000
    r = np.linspace(0.05, 1.0, num_stars)
    theta = r * spiral_tightness * np.pi + step * 0.3

    for arm in range(num_arms):
        arm_theta = theta + (2 * np.pi / num_arms) * arm
        x = r * np.cos(arm_theta) + np.random.normal(0, 0.02, num_stars)
        y = r * np.sin(arm_theta) + np.random.normal(0, 0.02, num_stars)
        color_shift = min(1.0, brightness)
        colors = plt.cm.plasma(np.linspace(0.2, color_shift, num_stars))
        ax.scatter(x, y, s=1.5, c=colors, alpha=0.9, edgecolors='none')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    plt.savefig(f"results/evolution_step_{step:02d}.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def simulate_evolution(pinn, vae, start_state, steps=15):
    t_vals = torch.linspace(0, 1, steps).unsqueeze(1).to(device)
    pinn.eval(); vae.eval()
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            t_input = t.expand(start_state.size(0), 1)
            evolved = pinn(start_state, t_input)
            draw_spiral_galaxy_features(i, evolved.cpu().numpy()[0])


def create_gif(folder="results", gif_name="Train_evolution.gif", duration=200):
    image_files = sorted(glob.glob(f"{folder}/evolution_step_*.png"))
    images = [Image.open(f) for f in image_files]
    if images:
        images[0].save(
            gif_name,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"🎞️ GIF saved as {gif_name}")
    else:
        print("⚠️ No evolution images found.")


def evaluate_vae(vae, data, threshold=0.5):
    """Evaluate VAE performance with multiple metrics"""
    vae.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = vae(data)
        
        # Calculate MSE and RMSE
        mse = nn.MSELoss()(reconstructed, data).item()
        rmse = np.sqrt(mse)
        
        # For classification metrics, binarize the data
        # Detach tensors before converting to numpy
        actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
        pred_binary = (reconstructed.detach().cpu().numpy() > threshold).astype(int)
        
        # Flatten for classification metrics
        actual_flat = actual_binary.flatten()
        pred_flat = pred_binary.flatten()
        
        # Calculate accuracy
        accuracy = accuracy_score(actual_flat, pred_flat)
        
        # Calculate F1 Score (micro-averaged over all features)
        f1 = f1_score(actual_flat, pred_flat, average='micro')
        
        # Create confusion matrix
        cm = confusion_matrix(actual_flat, pred_flat)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('VAE Reconstruction Confusion Matrix')
        plt.ylabel('True Values')
        plt.xlabel('Predicted Values')
        plt.savefig("results/vae_confusion_matrix.png")
        plt.close()
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Confusion Matrix': cm
        }


def evaluate_pinn(pinn, data, t_values=None):
    """Evaluate PINN performance with multiple metrics"""
    pinn.eval()
    
    if t_values is None:
        # Use a range of time values
        t_values = torch.linspace(0, 1, 10).to(device)
    
    results = []
    threshold = 0.5
    
    with torch.no_grad():
        for t in t_values:
            # Create t tensor for batch
            t_tensor = t.expand(data.size(0), 1)
            
            # Get predictions
            predictions = pinn(data, t_tensor)
            
            # Calculate MSE
            mse = nn.MSELoss()(predictions, data).item()
            rmse = np.sqrt(mse)
            
            # For classification metrics, binarize
            # Detach tensors before converting to numpy
            actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
            pred_binary = (predictions.detach().cpu().numpy() > threshold).astype(int)
            
            # Flatten for classification metrics
            actual_flat = actual_binary.flatten()
            pred_flat = pred_binary.flatten()
            
            # Calculate accuracy
            accuracy = accuracy_score(actual_flat, pred_flat)
            
            # Calculate F1 Score
            f1 = f1_score(actual_flat, pred_flat, average='micro')
            
            results.append({
                't': t.item(),
                'MSE': mse,
                'RMSE': rmse,
                'Accuracy': accuracy,
                'F1 Score': f1
            })
    
    # Calculate average metrics across time steps
    avg_mse = np.mean([r['MSE'] for r in results])
    avg_rmse = np.mean([r['RMSE'] for r in results])
    avg_accuracy = np.mean([r['Accuracy'] for r in results])
    avg_f1 = np.mean([r['F1 Score'] for r in results])
    
    # Create confusion matrix for middle time step for visualization
    mid_t = t_values[len(t_values)//2]
    mid_t_tensor = mid_t.expand(data.size(0), 1)
    mid_predictions = pinn(data, mid_t_tensor)
    
    # Detach tensors before converting to numpy
    mid_actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
    mid_pred_binary = (mid_predictions.detach().cpu().numpy() > threshold).astype(int)
    
    cm = confusion_matrix(mid_actual_binary.flatten(), mid_pred_binary.flatten())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'PINN Prediction Confusion Matrix (t={mid_t.item():.2f})')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.savefig("results/pinn_confusion_matrix.png")
    plt.close()
    
    return {
        'Detail': results,
        'Average MSE': avg_mse,
        'Average RMSE': avg_rmse,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Mid-t Confusion Matrix': cm
    }


def plot_metrics_over_time(pinn_results):
    """Plot PINN metrics over time"""
    metrics = ['MSE', 'RMSE', 'Accuracy', 'F1 Score']
    t_values = [r['t'] for r in pinn_results['Detail']]
    
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [r[metric] for r in pinn_results['Detail']]
        plt.plot(t_values, values, 'o-')
        plt.title(f'PINN {metric} vs Time')
        plt.xlabel('Time (t)')
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/pinn_metrics_over_time.png")
    plt.close()


if __name__ == "__main__":
    # ==== Train VAE ====
    vae = GalaxyVAE().to(device)
    vae_optim = optim.Adam(vae.parameters(), lr=0.001)

    def vae_loss(x, x_hat, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return recon + kl

    print("🔄 Training VAE...")
    vae.train()
    for epoch in range(240):
        vae_optim.zero_grad()
        x_hat, mu, logvar = vae(X_tensor)
        loss = vae_loss(X_tensor, x_hat, mu, logvar)
        loss.backward()
        vae_optim.step()
        print(f"VAE Epoch {epoch+1}, Loss: {loss.item():.2f}")
    torch.save(vae.state_dict(), "models/vae_galaxy.pth")
    print("✅ VAE model saved!")

    # ==== Train PINN ====
    pinn = GalaxyPINN().to(device)
    pinn_optim = optim.Adam(pinn.parameters(), lr=0.001)
    X_tensor.requires_grad = True
    timesteps = torch.linspace(0, 1, X_tensor.size(0)).unsqueeze(1).to(device).requires_grad_()

    print("🔄 Training PINN...")
    for epoch in range(240):
        pinn_optim.zero_grad()
        out = pinn(X_tensor, timesteps)
        loss = nn.MSELoss()(out, X_tensor) + physics_loss(out, timesteps)
        loss.backward()
        pinn_optim.step()
        print(f"PINN Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(pinn.state_dict(), "models/pinn_galaxy.pth")
    print("✅ PINN model saved!")

    # ==== Evaluate VAE Performance ====
    print("\n📊 Evaluating VAE Performance...")
    vae_metrics = evaluate_vae(vae, X_tensor)
    print(f"VAE MSE: {vae_metrics['MSE']:.4f}")
    print(f"VAE RMSE: {vae_metrics['RMSE']:.4f}")
    print(f"VAE Accuracy: {vae_metrics['Accuracy']:.4f}")
    print(f"VAE F1 Score: {vae_metrics['F1 Score']:.4f}")
    print(f"VAE Confusion Matrix:\n{vae_metrics['Confusion Matrix']}")
    print("✅ VAE confusion matrix saved to results/vae_confusion_matrix.png")

    # ==== Evaluate PINN Performance ====
    print("\n📊 Evaluating PINN Performance...")
    test_times = torch.linspace(0, 1, 10).to(device)
    pinn_metrics = evaluate_pinn(pinn, X_tensor, test_times)
    print(f"PINN Average MSE: {pinn_metrics['Average MSE']:.4f}")
    print(f"PINN Average RMSE: {pinn_metrics['Average RMSE']:.4f}")
    print(f"PINN Average Accuracy: {pinn_metrics['Average Accuracy']:.4f}")
    print(f"PINN Average F1 Score: {pinn_metrics['Average F1 Score']:.4f}")
    print("✅ PINN confusion matrix saved to results/pinn_confusion_matrix.png")
    
    # Plot PINN metrics over time
    plot_metrics_over_time(pinn_metrics)
    print("✅ PINN metrics over time plot saved to results/pinn_metrics_over_time.png")

    # ==== Select a Random Galaxy ====
    rand_idx = random.choice([0, 10, 100, 200, 500, 800])
    print(f"\n🌌 Simulating galaxy at index {rand_idx}")
    sample = X_tensor[rand_idx].unsqueeze(0)

    # ==== Simulate and Generate GIF ====
    simulate_evolution(pinn, vae, sample)
    create_gif()

    print("\n🌠 Done. You can now try different galaxies like:")
    print("   👉 X_tensor[10], X_tensor[100], X_tensor[500] and rerun the simulation.")