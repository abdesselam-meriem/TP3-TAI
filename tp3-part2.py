import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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

print("\n--- Partie 2: Convolution---")

def convolutionN(img, kernel):
    """
    Implémentation manuelle de la convolution avec padding
    """
    # Vérifier si l'image est en couleur ou niveaux de gris
    if len(img.shape) == 3:
        h, w, c = img.shape
        is_color = True
    else:
        h, w = img.shape
        is_color = False
        c = 1
    
    kh, kw = kernel.shape
    
    # CORRECTION: Inverser le noyau (nécessaire pour la convolution)
    kernel = np.flip(kernel, axis=[0, 1])
    
    # Calculer le padding
    p = (kh - 1) // 2
    
    # Appliquer le padding
    if is_color:
        padded = np.zeros((h + 2*p, w + 2*p, c), dtype=img.dtype)
        for channel in range(c):
            padded[:, :, channel] = np.pad(img[:, :, channel], ((p, p), (p, p)), 
                                         mode='constant', constant_values=0)
    else:
        padded = np.pad(img, ((p, p), (p, p)), mode='constant', constant_values=0)
    
    # Initialiser l'image de sortie
    if is_color:
        output = np.zeros((h, w, c), dtype=float)
    else:
        output = np.zeros((h, w), dtype=float)
    
    # Appliquer la convolution
    for i in range(h):
        for j in range(w):
            if is_color:
                for channel in range(c):
                    region = padded[i:i+kh, j:j+kw, channel]
                    # CORRECTION: Utiliser np.sum() au lieu de multiplication simple
                    output[i, j, channel] = np.sum(region * kernel)
            else:
                region = padded[i:i+kh, j:j+kw]
                output[i, j] = np.sum(region * kernel)
    
    return output

# Tester la fonction de convolution
print("Test de la fonction de convolution personnalisée...")
kernel_test = np.array([[1, 0, -1], 
                       [0, 1, 0], 
                       [1, 0, -1]], dtype=np.float32)

# Appliquer la convolution
result_custom = convolutionN(img_gray, kernel_test)

# Normaliser pour l'affichage
result_normalized = cv2.normalize(result_custom, None, 0, 255, cv2.NORM_MINMAX)
result_uint8 = np.uint8(result_normalized)

# Comparer avec OpenCV (pour vérification)
result_opencv = cv2.filter2D(img_gray, -1, kernel_test)

# Afficher les résultats
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(result_custom, cmap='gray')
plt.title('Notre Convolution')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(result_uint8, cmap='gray')
plt.title('Résultat Normalisé')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(result_opencv, cmap='gray')
plt.title('OpenCV filter2D')
plt.axis('off')

plt.tight_layout()
plt.show()
