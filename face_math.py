import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from PIL import Image
import os

# --- STEP 1: LOAD THE "KNOWLEDGE BASE" ---
# We use 400 existing faces to teach the computer what a "human face" looks like.
print("Step 1: Teaching the computer the 'average' human features...")
faces_data = fetch_olivetti_faces()
faces = faces_data.data

# PCA (Principal Component Analysis) is our Math Tool.
# It finds the 50 most important patterns (Eigenfaces) shared by all humans.
n_components = 50
pca = PCA(n_components=n_components, whiten=True).fit(faces)

# --- STEP 2: GET THE NEW PHOTO ---
# We need a new face to test our math on.
path = input("\n[ACTION] Enter the path to your image (e.g., C:/photo.jpg): ").strip()

if os.path.exists(path):
    # --- STEP 3: PRE-PROCESSING (The "Cleanup") ---
    # The math only works if your photo looks like the training data.
    # We turn it to Grayscale ('L') and resize it to a tiny 64x64 grid.
    img = Image.open(path).convert('L').resize((64, 64))

    # Turn the image into a list of numbers (Pixels) and normalize them (0 to 1)
    img_array = np.array(img) / 255.0

    # Flatten: Turn a 64x64 square into one long line of 4,096 numbers.
    # In math, this is your "Face Vector."
    img_vector = img_array.reshape(1, -1)

    # --- STEP 4: THE PROJECTION (The "Translation") ---
    # We ask: "How much of each Eigenface do we need to build THIS specific face?"
    # It turns 4,096 pixels into just 50 "Weight" numbers.
    weights = pca.transform(img_vector)

    # --- STEP 5: THE RECONSTRUCTION (The "Math Ghost") ---
    # Now we try to rebuild your face using ONLY those 50 weights.
    # Formula: Face = (Weight1 * Eigenface1) + (Weight2 * Eigenface2) ...
    reconstruction = pca.inverse_transform(weights).reshape(64, 64)

    # --- STEP 6: COMPARE ---
    # We show the original vs. what the math "remembered."
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title("What You See (Original)")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstruction, cmap='gray')
    plt.title(f"What Math Sees ({n_components} Key Features)")

    plt.show()

    print(f"\nSuccess! Your face is now stored as just {n_components} coordinates.")
else:
    print("Error: I couldn't find that file. Double-check the path!")