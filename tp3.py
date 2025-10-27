# TAL_TP3.py - Script complet
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

print("=== TP3 - Filtres et Convolutions ===")

# =============================================================================
# PARTIE 1: Convolution avec OpenCV
# =============================================================================
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
custom_kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)

# Appliquer le noyau personnalisé
custom_filtered = cv2.filter2D(img_rgb, -1, custom_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(custom_filtered)
plt.title('Filtre Personnalisé (Détection contours)')
plt.axis('off')
plt.tight_layout()
plt.show()

print("✓ Partie 1 terminée avec succès")

# =============================================================================
# PARTIE 2: Convolution "from scratch"
# =============================================================================
print("\n--- Partie 2: Convolution from scratch ---")

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

print("✓ Partie 2 terminée avec succès")

# =============================================================================
# PARTIE 3: Filtre Gaussien
# =============================================================================
print("\n--- Partie 3: Filtre Gaussien ---")

def filtre_Gaussien(sigma, k_size):
    """
    Génère un noyau gaussien
    sigma: écart-type de la distribution gaussienne
    k_size: taille du noyau (doit être impair)
    """
    # Vérifier que la taille est impaire
    if k_size % 2 == 0:
        k_size += 1
        print(f"Attention: Taille du noyau ajustée à {k_size} (doit être impaire)")
    
    # Créer une grille de coordonnées
    k_radius = k_size // 2
    x = np.arange(-k_radius, k_radius + 1)
    y = np.arange(-k_radius, k_radius + 1)
    X, Y = np.meshgrid(x, y)
    
    # Calculer la distribution gaussienne 2D
    gaussian = (1 / (2 * math.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Normaliser le noyau (somme = 1)
    gaussian = gaussian / np.sum(gaussian)
    
    return gaussian

# 1. Afficher les valeurs pour K=3 et σ=1
print("1. Noyau Gaussien pour K=3, σ=1:")
kernel_gauss_3x3 = filtre_Gaussien(sigma=1, k_size=3)
print(kernel_gauss_3x3)

# 2. Tester différentes configurations
print("\n2. Test de différentes configurations gaussiennes...")
configurations = [
    (3, 0.5), (3, 1), (3, 2),
    (5, 1), (7, 1), (11, 1)
]

plt.figure(figsize=(20, 10))

# Afficher l'image originale
plt.subplot(2, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Image Originale (Bruitée)')
plt.axis('off')

# Appliquer et afficher les différents filtres gaussiens
for i, (k_size, sigma) in enumerate(configurations, 2):
    # Générer le noyau gaussien
    kernel = filtre_Gaussien(sigma, k_size)
    
    # Appliquer la convolution avec notre fonction
    result = convolutionN(img_gray, kernel)
    
    # Normaliser
    result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result_uint8 = np.uint8(result_normalized)
    
    plt.subplot(2, 4, i)
    plt.imshow(result_uint8, cmap='gray')
    plt.title(f'Gaussien {k_size}x{k_size}\nσ={sigma}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 3. Comparaison avec la fonction OpenCV
print("\n3. Comparaison avec OpenCV...")

# Configuration de test
k_size_test = 5
sigma_test = 1.5

# Notre implémentation
our_kernel = filtre_Gaussien(sigma_test, k_size_test)
our_result = convolutionN(img_gray, our_kernel)
our_result_normalized = cv2.normalize(our_result, None, 0, 255, cv2.NORM_MINMAX)
our_result_uint8 = np.uint8(our_result_normalized)

# OpenCV
opencv_result = cv2.GaussianBlur(img_gray, (k_size_test, k_size_test), sigma_test)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Originale')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(our_result_uint8, cmap='gray')
plt.title('Notre Implémentation')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(opencv_result, cmap='gray')
plt.title('OpenCV GaussianBlur')
plt.axis('off')

# Différence
difference = cv2.absdiff(our_result_uint8, opencv_result)
plt.subplot(1, 4, 4)
plt.imshow(difference, cmap='hot')
plt.title('Différence')
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"Différence moyenne entre les deux implémentations: {np.mean(difference):.2f}")

# 4. Démonstration des effets de sigma
print("\n4. Effet du paramètre sigma...")

k_size_fixed = 7
sigmas = [0.5, 1, 2, 4]

plt.figure(figsize=(15, 10))

for i, sigma in enumerate(sigmas, 1):
    # Générer et appliquer le filtre
    kernel = filtre_Gaussien(sigma, k_size_fixed)
    result = convolutionN(img_gray, kernel)
    result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result_uint8 = np.uint8(result_normalized)
    
    plt.subplot(2, 4, i)
    plt.imshow(result_uint8, cmap='gray')
    plt.title(f'σ = {sigma}')
    plt.axis('off')
    
    # Afficher le noyau
    plt.subplot(2, 4, i + 4)
    plt.imshow(kernel, cmap='hot', interpolation='nearest')
    plt.title(f'Noyau σ={sigma}')
    plt.axis('off')
    plt.colorbar()

plt.tight_layout()
plt.show()

print("✓ Partie 3 terminée avec succès")

# =============================================================================
# RÉSUMÉ ET CONCLUSIONS
# =============================================================================
print("\n" + "="*50)
print("RÉSUMÉ DU TP3")
print("="*50)

print("\n📊 Ce TP vous a permis de:")
print("✓ Comprendre le principe de la convolution d'image")
print("✓ Explorer les fonctions de convolution d'OpenCV")
print("✓ Appliquer différents filtres (moyen, médian, gaussien)")
print("✓ Implémenter votre propre fonction de convolution")
print("✓ Comprendre l'importance du padding et de l'inversion du noyau")
print("✓ Générer et appliquer des filtres gaussiens")

print("\n🔍 Observations importantes:")
print("- Le filtre moyen est simple mais peut créer un flou excessif")
print("- Le filtre médian est efficace contre le bruit 'sel et poivre'")
print("- Le filtre gaussien procure un flou plus naturel")
print("- Le paramètre σ contrôle l'étalement du flou gaussien")
print("- L'inversion du noyau est cruciale pour une convolution correcte")

print("\n🎯 Conseils pour la suite:")
print("- Expérimentez avec d'autres types de noyaux (Sobel, Laplacien, etc.)")
print("- Testez l'effet de différents types de padding (reflect, edge, etc.)")
print("- Explorez l'impact de la convolution sur la détection de contours")

print("\n✅ TP3 terminé avec succès!")