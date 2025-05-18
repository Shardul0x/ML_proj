# test.py — Realistic Spiral Galaxy Evolution Based on Metadata

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import argparse
from matplotlib.colors import LinearSegmentedColormap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_folder = "test_results"
os.makedirs(result_folder, exist_ok=True)


# Define the VAE and PINN models (copied from training code)
class ImprovedGalaxyVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[32, 16], latent_dim=5):
        super(ImprovedGalaxyVAE, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse the hidden dimensions
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        # Final output layer
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


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


def create_custom_cmap(features=None):
    """Create a custom colormap for galaxy visualization to match the sample image
    
    Parameters:
    - features: optional array of galaxy features to influence color scheme
    """
    # Default color palette to match the sample image (yellow outer, transitioning to purple/blue center)
    if features is None or features[0] < 0.5:  # Spiral galaxy - use yellow to purple
        colors = [(0.95, 0.95, 0.2),    # Yellow (outer)
                 (0.9, 0.6, 0.2),      # Orange
                 (0.9, 0.1, 0.6),      # Pink
                 (0.7, 0.0, 0.9),      # Purple 
                 (0.4, 0.2, 0.8)]      # Blue-purple (center)
    else:  # Elliptical galaxy - use more reddish to blue
        colors = [(0.9, 0.8, 0.7),     # Light yellow (outer)
                 (0.9, 0.5, 0.3),      # Orange
                 (0.8, 0.3, 0.3),      # Reddish
                 (0.6, 0.2, 0.5),      # Purplish
                 (0.3, 0.2, 0.6)]      # Bluish (center)

    # If features are provided, adjust colors based on galaxy properties
    if features is not None:
        P_EL, P_CW, P_ACW, P_EDGE, P_DK, P_MG, P_CS, P_EL_DEBIASED, P_CS_DEBIASED = features
        
        # More dust means more reddish
        if P_DK > 0.5:
            colors[1] = (0.9, 0.4, 0.1)  # More reddish
            colors[2] = (0.8, 0.2, 0.2)  # More reddish
        
        # Mergers have more irregular color patterns
        if P_MG > 0.5:
            colors[2] = (0.7, 0.3, 0.7)  # More purple
            colors[3] = (0.5, 0.3, 0.7)  # More blue-purple
            
        # Higher central concentration means more intense core colors
        if P_CS > 0.7:
            colors[4] = (0.3, 0.1, 0.9)  # More intense blue-purple

    # Create custom colormap
    return LinearSegmentedColormap.from_list('galaxy_cmap', colors, N=256)


def draw_realistic_galaxy(step, features, save_path, visual_style='auto', seed=None):
    """
    Draw a realistic galaxy based on evolved features
    
    Parameters:
    - step: current timestep in the evolution
    - features: array of galaxy features
    - save_path: where to save the image
    - visual_style: 'spiral', 'elliptical', or 'auto'
    - seed: random seed for reproducibility
    """
    # Set random seed if provided (for consistent galaxy patterns across runs)
    if seed is not None:
        np.random.seed(seed + step)
    
    P_EL, P_CW, P_ACW, P_EDGE, P_DK, P_MG, P_CS, P_EL_DEBIASED, P_CS_DEBIASED = features
    
    # Determine visual style based on P_EL if set to auto
    if visual_style == 'auto':
        visual_style = 'elliptical' if P_EL > 0.5 else 'spiral'
    
    # Higher DPI for better quality
    fig = plt.figure(figsize=(10, 10), dpi=250, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    
    # Create custom colormap based on galaxy features
    galaxy_cmap = create_custom_cmap(features)
    
    if visual_style == 'elliptical':
        # === ELLIPTICAL GALAXY VISUALIZATION ===
        # Parameters directly based on features
        brightness = 0.5 + P_CS * 0.5
        size = 2.0 - 0.5 * P_EDGE  # Smaller for edge-on
        elongation = 1.0 - 0.6 * P_EDGE  # More elongated for edge-on
        
        # Adjust concentration based on debiased central concentration
        concentration = 2.0 + 3.0 * P_CS_DEBIASED
        
        # Merger effect - makes the galaxy more asymmetric
        asymmetry = P_MG * 0.5
        
        # Draw the elliptical galaxy
        num_points = int(30000 + 20000 * P_CS)  # More points for higher concentration
        
        # Generate random points in a 2D Gaussian distribution
        x = np.random.normal(0, size, num_points)
        y = np.random.normal(0, size * elongation, num_points)
        
        # Add asymmetry if merger
        if asymmetry > 0.1:
            # Add some offset to create asymmetric shape
            offset_x = asymmetry * np.random.normal(0, 1)
            offset_y = asymmetry * np.random.normal(0, 1)
            mask = np.random.random(num_points) < 0.3
            x[mask] += offset_x
            y[mask] += offset_y
        
        # Apply de Vaucouleurs' R^(1/4) law for surface brightness
        r = np.sqrt(x**2 + y**2)
        brightness_scale = np.exp(-concentration * (r**(1/4) - 1))
        brightness_cutoff = 0.05
        valid_points = brightness_scale > brightness_cutoff
        
        x = x[valid_points]
        y = y[valid_points]
        brightness_scale = brightness_scale[valid_points]
        
        # Rotate the galaxy - rotation speed depends on P_CW and P_ACW
        rotation_speed = 0.02 * max(P_CW, P_ACW)
        theta = step * rotation_speed
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        
        # Use the custom colormap
        point_colors = galaxy_cmap(1.0 - brightness_scale)
        
        # Draw with varying brightness
        sizes = brightness_scale * (0.6 + 0.4 * P_CS)  # Point size affected by concentration
        alpha = np.clip(brightness_scale * 0.8, 0, 1)
        
        ax.scatter(x_rot, y_rot, s=sizes, c=point_colors, alpha=alpha)
        
        # Add more concentrated core
        core_points = int(num_points * 0.2 * P_CS)
        core_x = np.random.normal(0, size * 0.2, core_points)
        core_y = np.random.normal(0, size * 0.2 * elongation, core_points)
        core_brightness = np.random.power(0.5, core_points)
        
        # Core color is affected by P_CS_DEBIASED
        core_colors = galaxy_cmap(0.7 + 0.3 * P_CS_DEBIASED * np.ones(core_points))
        
        ax.scatter(core_x, core_y, s=core_brightness * 0.8, 
                   c=core_colors, 
                   alpha=np.clip(core_brightness, 0, 1))
        
    else:
        # === SPIRAL GALAXY VISUALIZATION ===
        # Parameters directly based on features
        
        # Number of arms determined by clockwise and counterclockwise probabilities
        # For the sample image, use 2 arms typically
        if P_CW > 0.7 or P_ACW > 0.7:
            num_arms = 2  # Strong spiral pattern
        elif P_CW > 0.4 or P_ACW > 0.4:
            num_arms = max(2, min(5, int(2 + round(3 * max(P_CW, P_ACW)))))
        else:
            num_arms = 2  # Default to 2 arms
            
        # Arm tightness (higher value = more tightly wound)
        arm_tightness = 3.0 + 6.0 * (1 - P_EL)
        
        # Dust lanes based on P_DK
        dust_lanes = P_DK > 0.3
        dust_amount = P_DK * 0.8
        
        # Merger effect creates disturbed arms
        arm_disturbance = P_MG * 0.5
        
        # Generate stars for the central bulge
        bulge_size = 0.2 + 0.3 * P_CS_DEBIASED
        bulge_points = int(20000 * P_CS)
        
        if bulge_points > 0:
            # Central bulge with de Vaucouleurs profile
            r_bulge = np.random.power(0.25, bulge_points) * bulge_size
            theta_bulge = np.random.uniform(0, 2*np.pi, bulge_points)
            x_bulge = r_bulge * np.cos(theta_bulge)
            y_bulge = r_bulge * np.sin(theta_bulge)
            
            # Color the bulge using the custom colormap
            # Bulge color depends on P_CS_DEBIASED - higher values make it more yellow/red
            # Map bulge brightness to positions on the colormap
            bulge_pos = 0.6 + 0.4 * P_CS_DEBIASED
            bulge_colors = galaxy_cmap(bulge_pos + 0.1 * np.random.random(bulge_points))
            
            # Use smaller point sizes for more realism
            bulge_sizes = 0.4 + 0.4 * P_CS  # Larger points for higher concentration
            ax.scatter(x_bulge, y_bulge, s=bulge_sizes, c=bulge_colors)
        
        # Generate the spiral arms
        points_per_arm = int(8000 + 4000 * (P_CW + P_ACW))  # More points for stronger spiral
        
        for arm in range(num_arms):
            # Base spiral
            r = np.linspace(0.2, 1.6, points_per_arm)
            
            # Rotation for this arm + animation rotation
            # Rotation direction based on P_CW and P_ACW
            base_angle = (2 * np.pi / num_arms) * arm
            if P_CW > P_ACW:
                rotation = -step * 0.05  # Clockwise
            else:
                rotation = step * 0.05   # Anti-clockwise
                
            # Logarithmic spiral formula
            theta = base_angle + arm_tightness * np.log(r) + rotation
            
            # Add noise/disturbance to the arm - affected by merger probability
            theta_noise = (0.1 + 0.4 * arm_disturbance) * np.random.normal(0, 0.1, points_per_arm) * (r**0.5)
            theta += theta_noise
            
            # Calculate positions
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Add some width to the arms - affected by edge-on probability
            width_factor = 0.07 * (1.0 - 0.5 * P_EDGE)
            perpendicular_x = -np.sin(theta)
            perpendicular_y = np.cos(theta)
            
            width_variation = width_factor * r * np.random.normal(0, 1, points_per_arm)
            x += perpendicular_x * width_variation
            y += perpendicular_y * width_variation
            
            # Color gradient along the arm - map radius to position in custom colormap
            colors = galaxy_cmap(1.0 - r/1.6)
            
            # Adjust alpha for smoother appearance and based on edge-on probability
            alpha = np.clip(0.7 - 0.3 * (r/1.6) * (1.0 - 0.5 * P_EDGE), 0.4, 0.9)
            for i in range(len(colors)):
                colors[i][3] = alpha[i]
            
            # Draw arm with size affected by spiral strength
            sizes = (0.4 + 0.4 * max(P_CW, P_ACW)) + 0.2 * np.random.random(points_per_arm)
            ax.scatter(x, y, s=sizes, c=colors)
            
            # Add dust lanes if needed
            if dust_lanes:
                dust_points = int(points_per_arm * dust_amount)
                dust_indices = np.random.choice(points_per_arm, dust_points, replace=False)
                dust_x = x[dust_indices] + 0.02 * np.random.normal(0, 1, dust_points)
                dust_y = y[dust_indices] + 0.02 * np.random.normal(0, 1, dust_points)
                dust_alpha = np.clip(0.3 + 0.2 * np.random.random(dust_points), 0, 1)
                ax.scatter(dust_x, dust_y, s=0.6, color='black', alpha=dust_alpha)
    
    # Add a few background stars (just a few so they don't dominate)
    num_bg_stars = 200
    bg_x = np.random.uniform(-2, 2, num_bg_stars)
    bg_y = np.random.uniform(-2, 2, num_bg_stars)
    bg_sizes = 0.2 + 0.2 * np.random.power(2, num_bg_stars)
    ax.scatter(bg_x, bg_y, s=bg_sizes, color='black', alpha=0.5)
    
    # Set limits and remove axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white', dpi=250)
    plt.close()


def create_gif(frames, gif_name="galaxy_evolution.gif", duration=150):
    """Create a GIF from the generated frames"""
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"🎞️ GIF saved as {gif_name}")


def generate_example_metadata():
    """Generate a set of example galaxy metadata presets"""
    examples = {
        "sample_image": [0.1, 0.6, 0.3, 0.1, 0.1, 0.1, 0.5, 0.1, 0.5],  # Spiral galaxy like sample
        "spiral_galaxy": [0.2, 0.7, 0.2, 0.1, 0.1, 0.1, 0.8, 0.2, 0.8],  # Standard spiral galaxy
        "elliptical_galaxy": [0.9, 0.1, 0.1, 0.2, 0.2, 0.6, 0.5, 0.9, 0.5],  # Elliptical galaxy
        "barred_spiral": [0.3, 0.5, 0.4, 0.1, 0.6, 0.1, 0.9, 0.3, 0.9],  # Barred spiral galaxy
        "edge_on_galaxy": [0.2, 0.3, 0.3, 0.9, 0.5, 0.1, 0.7, 0.2, 0.7],  # Edge-on spiral
        "dusty_galaxy": [0.3, 0.4, 0.4, 0.3, 0.9, 0.2, 0.6, 0.3, 0.6],  # Dusty spiral galaxy
        "merger_galaxy": [0.4, 0.3, 0.3, 0.2, 0.4, 0.9, 0.5, 0.4, 0.5],  # Galaxy merger
        "faint_spiral": [0.2, 0.5, 0.3, 0.2, 0.2, 0.1, 0.3, 0.2, 0.3],  # Faint spiral with small bulge
    }
    return examples


def load_models_or_dummy():
    """Try to load models, or create dummy models if not found"""
    print(" Loading or creating models...")
    
    vae = ImprovedGalaxyVAE(input_dim=9, hidden_dims=[32, 16], latent_dim=5).to(device)
    pinn = GalaxyPINN(input_dim=9, hidden_dim=64).to(device)
    
    try:
        vae.load_state_dict(torch.load("models/vae_galaxy.pth", map_location=device))
        pinn.load_state_dict(torch.load("models/pinn_galaxy.pth", map_location=device))
        print("Models loaded successfully")
        return vae, pinn, False
    except FileNotFoundError:
        print(" Models not found. Creating dummy models for visualization only.")
        return vae, pinn, True


def get_galaxy_metadata(examples):
    """Get galaxy metadata from user input or use a preset example"""
    print("\n" + "="*60)
    print("Galaxy Feature Input Mode:")
    print("1. Enter custom values")
    choice = input("Press 1 to continue : ")
    
    # Map choices to example keys
    choice_map = {
        
    }
    
    if choice == "1":
        print("\nEnter values between 0.0 and 1.0 for each feature:")
        try:
            meta_input = []
            feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
            
            for col in feature_cols:
                desc = {
                    'P_EL': "Probability of being elliptical",
                    'P_CW': "Probability of having clockwise spiral arms",
                    'P_ACW': "Probability of having anti-clockwise spiral arms",
                    'P_EDGE': "Probability of being edge-on",
                    'P_DK': "Probability of having dust lanes",
                    'P_MG': "Probability of being a merger",
                    'P_CS': "Probability of having a central bulge",
                    'P_EL_DEBIASED': "Debiased probability of being elliptical",
                    'P_CS_DEBIASED': "Debiased probability of having a central bulge"
                }
                
                value = float(input(f"{col} ({desc[col]}): "))
                if not 0 <= value <= 1:
                    print(f" Value must be between 0.0 and 1.0. Using 0.5 for {col}.")
                    value = 0.5
                meta_input.append(value)
        except ValueError:
            print(" Invalid input. Using sample image example instead.")
            meta_input = examples["sample_image"]
    elif choice in choice_map:
        example_key = choice_map[choice]
        print(f"\nUsing preset {example_key.replace('_', ' ')} example")
        meta_input = examples[example_key]
    else:
        print("\nInvalid choice. Using sample image example instead.")
        meta_input = examples["sample_image"]
    
    print("\nFeature values:")
    feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {meta_input[i]:.2f}")
    
    return meta_input


def dummy_evolution(initial_features, steps=60):
    """Create dummy evolution pattern when models aren't available"""
    print(f"\n Creating galaxy evolution pattern over {steps} timesteps...")
    
    evolved_features = []
    
    # Make a copy of the initial features
    features = np.array(initial_features)
    
    # Generate time evolution with some patterns
    for i in range(steps):
        t = i / steps  # Normalized time
        
        # Create a copy of current features
        current = features.copy()
        
        # Apply some evolution patterns:
        
        # 1. Gradually increase concentration over time
        current[6] = min(1.0, features[6] + 0.2 * t)  # P_CS increases
        current[8] = min(1.0, features[8] + 0.2 * t)  # P_CS_DEBIASED increases
        
        # 2. Spiral arms fade as elliptical increases slightly
        if features[0] < 0.6:  # If not already strongly elliptical
            current[0] = min(0.6, features[0] + 0.3 * t)  # P_EL increases
            current[7] = min(0.6, features[7] + 0.3 * t)  # P_EL_DEBIASED increases
            
            # As spiral becomes more elliptical, reduce spiral strengths
            current[1] = max(0.1, features[1] - 0.2 * t)  # P_CW decreases
            current[2] = max(0.1, features[2] - 0.2 * t)  # P_ACW decreases
        
        # 3. Dust lanes may increase then decrease
        dust_pattern = np.sin(t * np.pi) * 0.3
        current[4] = max(0.0, min(1.0, features[4] + dust_pattern))  # P_DK changes
        
        # 4. Edge-on view might change (viewing angle)
        if i % (steps // 4) == 0:  # Change direction occasionally
            current[3] = min(0.9, max(0.1, features[3] + np.random.uniform(-0.2, 0.2)))
            
        # 5. Merger probability may have a small spike in the middle of the evolution
        if 0.4 < t < 0.6:
            current[5] = min(0.8, features[5] + 0.3)  # P_MG increases temporarily
        else:
            current[5] = features[5]
            
        evolved_features.append(current)
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{steps} timesteps")
    
    return evolved_features


def simulate_evolution(vae, pinn, sample_tensor, steps=30, use_dummy=False, initial_features=None):
    """Simulate galaxy evolution using the PINN model or dummy evolution if models not available"""
    if use_dummy and initial_features is not None:
        return dummy_evolution(initial_features, steps)
    
    print(f"\n Simulating galaxy evolution over {steps} timesteps...")
    
    t_vals = torch.linspace(0, 1, steps).unsqueeze(1).to(device)
    evolved_features = []
    
    vae.eval()
    pinn.eval()
    
    with torch.no_grad():
        # Generate time evolution
        for i, t in enumerate(t_vals):
            t_input = t.expand(sample_tensor.size(0), 1)
            evolved = pinn(sample_tensor, t_input)
            evolved_features.append(evolved.cpu().numpy()[0])
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{steps} timesteps")
    
    return evolved_features


def main():
    parser = argparse.ArgumentParser(description="Galaxy Evolution Simulator")
    parser.add_argument("--steps", type=int, default=60, help="Number of evolution steps")
    parser.add_argument("--style", type=str, default="auto", choices=["auto", "spiral", "elliptical"], 
                        help="Visualization style (auto determines based on P_EL)")
    parser.add_argument("--duration", type=int, default=120, help="GIF frame duration in ms")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set master random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("\nGalaxy Evolution Simulator ")
    print("This tool simulates the evolution of galaxies based on metadata features.")
    
    # Get example metadata options
    examples = generate_example_metadata()
    
    # Load or create models
    vae, pinn, use_dummy = load_models_or_dummy()
    
    # Get metadata
    meta_input = get_galaxy_metadata(examples)
    
    # Try to load reference dataset for scaling
    try:
        df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
        feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
                        'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
        df = df[feature_cols].dropna()
        print(" Reference dataset loaded for feature scaling")
        
        # Scale input
        scaler = MinMaxScaler()
        scaler.fit(df)
        sample_scaled = scaler.transform([meta_input])
    except FileNotFoundError:
        print(" Dataset file not found. Using unscaled input.")
        sample_scaled = np.array([meta_input])
    except Exception as e:
        print(f" Error loading dataset: {e}")
        sample_scaled = np.array([meta_input])
    
    # Convert to tensor
    sample_tensor = torch.FloatTensor(sample_scaled).to(device)
    
    # Run simulation - either with model or dummy evolution
    evolved_features = simulate_evolution(vae, pinn, sample_tensor, 
                                         steps=args.steps, 
                                         use_dummy=use_dummy,
                                         initial_features=meta_input)
    
    print(f"\n Generating galaxy visualizations for {args.steps} timesteps...")
    
    # Create frames folder if it doesn't exist
    frames_folder = os.path.join(result_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Generate frames
    frames = []
    for step, features in enumerate(evolved_features):
        # Ensure features are within valid range [0, 1]
        features = np.clip(features, 0, 1)
        
        # Generate image
        frame_path = os.path.join(frames_folder, f"frame_{step:03d}.png")
        print(f"  Generating frame {step+1}/{args.steps}", end="\r")
        
        # Draw the galaxy
        draw_realistic_galaxy(step, features, frame_path, 
                             visual_style=args.style,
                             seed=args.seed)
        
        # Load image for GIF creation
        img = Image.open(frame_path)
        frames.append(img)
    
    print("\n All frames generated")
    
    # Create GIF
    gif_path = os.path.join(result_folder, "galaxy_evolution.gif")
    create_gif(frames, gif_path, duration=args.duration)
    
    # Display summary and instructions
    print("\nGalaxy evolution simulation complete!")
    print(f"• {args.steps} evolution steps generated")
    print(f"• Individual frames saved in: {frames_folder}")
    print(f"• Animation saved as: {gif_path}")
    print("\nTo view another evolution, run the script again.")


if __name__ == "__main__":
    main()