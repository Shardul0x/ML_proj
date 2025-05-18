# === Galaxy Evolution Simulator ===
# Uses Galaxy Zoo images + metadata to simulate galaxy formation
# Combines VAE + Physics-Informed Network + 3D Interactive Visualization

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Model, backend as K
import tensorflow as tf
import pyvista as pv

# === Step 1: Preprocessing ===
CSV_PATH = "GalaxyZoo1_DR_table2.csv"  # Your Galaxy Zoo CSV
IMAGE_FOLDER = "C:/Users/ASUS/Downloads/ML/images_gz2/images"  # Folder with galaxy images
IMG_SIZE = 64

# Load and clean data
df = pd.read_csv(CSV_PATH).dropna()

# Get available image filenames without extension
available_imgs = set([f.split('.')[0] for f in os.listdir(IMAGE_FOLDER)])

# Filter the DataFrame to include only rows with matching OBJID values
df = df[df['OBJID'].astype(str).isin(available_imgs)]

def load_data(df, folder):
    images, metadata = [], []
    for _, row in df.iterrows():
        objid = str(row['OBJID'])
        path = os.path.join(folder, objid + ".jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            images.append(img)
            metadata.append([
                row['P_EL'], row['P_CW'], row['P_ACW'],
                row['P_EDGE'], row['P_MG'], row['P_CS']
            ])
    return np.array(images), np.array(metadata)

X_img, X_meta = load_data(df, IMAGE_FOLDER)

# Check if images were loaded
if len(X_img) == 0:
    raise ValueError("❌ No images were loaded. Check your CSV/filenames/folder path.")

# Train-test split
X_train_img, X_test_img, X_train_meta, X_test_meta = train_test_split(X_img, X_meta, test_size=0.2)

# === Step 2: Variational Autoencoder ===
latent_dim = 64

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Encoder
img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(img_input)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = Model(img_input, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(16 * 16 * 64, activation='relu')(latent_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
outputs = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

# VAE Model
vae_input = img_input
z_mean, z_log_var, z = encoder(vae_input)
vae_output = decoder(z)
vae = Model(vae_input, vae_output, name='vae')

# VAE Loss
reconstruction_loss = tf.keras.losses.MeanSquaredError()(vae_input, vae_output)
kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
vae.add_loss(reconstruction_loss + kl_loss)
vae.compile(optimizer='adam')

# Train VAE
vae.fit(X_train_img, X_train_img, epochs=20, batch_size=32, validation_data=(X_test_img, X_test_img))

# === Step 3: PINN (Predict future latent z) ===
pinn = models.Sequential([
    layers.Input(shape=(6,)),  # 6 metadata features
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(latent_dim)
])
pinn.compile(optimizer='adam', loss='mse')

# Generate training z from encoder
z_train = encoder.predict(X_train_img)[2]
pinn.fit(X_train_meta, z_train, epochs=30, batch_size=32)

# === Step 4: Galaxy Evolution and 3D Visualization ===
def simulate_evolution(metadata_sequence):
    frames = []
    for meta in metadata_sequence:
        z = pinn.predict(np.array([meta]))
        img = decoder.predict(z)
        frames.append(img[0])
    return frames

def visualize_3d(frames):
    plotter = pv.Plotter()
    for i, frame in enumerate(frames):
        intensity = frame.mean(axis=2)
        grid = pv.UniformGrid(
            dims=(IMG_SIZE, IMG_SIZE, 1),
            spacing=(1, 1, 1),
            origin=(0, 0, i)
        )
        grid.point_arrays['intensity'] = intensity.flatten(order="F")
        plotter.add_volume(grid, cmap="plasma", opacity="sigmoid")
    plotter.show()

# Example evolution (simulate 10 time steps)
example = X_test_meta[0]
sequence = [example + i * 0.01 for i in range(10)]  # simulated gradual change
frames = simulate_evolution(sequence)
visualize_3d(frames)
