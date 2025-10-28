import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

print("\n--- Partie 1: Convolution avec OpenCV ---")

# Charger l'image
img = cv2.imread('lena_noise.jpg')
if img is None:
    print("Erreur: Impossible de charger l'image 'lena_noise.jpg'")
    print("Vérifiez que l'image est dans le même répertoire que le script")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Image chargée avec succès")
print(f"Dimensions: {img.shape}")

# 1. Filtre moyen avec différentes tailles de noyau
print("\n1. Application du filtre moyen...")
kernel_sizes = [3, 5, 7, 11]

plt.figure(figsize=(15, 10))

# Image originale
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Image Originale')
plt.axis('off')

# Filtre moyen
for i, k_size in enumerate(kernel_sizes, 2):
    # Créer le noyau de moyenne
    kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)
    
    # Appliquer la convolution
    filtered_img = cv2.filter2D(img_rgb, -1, kernel)
    
    plt.subplot(2, 3, i)
    plt.imshow(filtered_img)
    plt.title(f'Filtre Moyen {k_size}x{k_size}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2. Filtre médian avec différentes tailles de noyau
print("2. Application du filtre médian...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Image Originale')
plt.axis('off')

for i, k_size in enumerate(kernel_sizes, 2):
    # Appliquer le filtre médian
    filtered_img = cv2.medianBlur(img_rgb, k_size)
    
    plt.subplot(2, 3, i)
    plt.imshow(filtered_img)
    plt.title(f'Filtre Médian {k_size}x{k_size}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 3. Noyau personnalisé
print("3. Application d'un noyau personnalisé...")
custom_kernel = np.array([[0, 0, 0],
                          [0, 3, 0],
                          [0, 0, 0]], dtype=np.float32)

# Appliquer le noyau
custom_filtered = cv2.filter2D(img_rgb, -1, custom_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(custom_filtered)
plt.title('Filtre Personnalisé cnetre 3')
plt.axis('off')
plt.tight_layout()
plt.show()
