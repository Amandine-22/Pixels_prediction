#Import des modules 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import os

#================================================================
# Predictions d'un unique pixels 
#================================================================

#Importation d'une image 
path = '/Users/amandine/Desktop/Perso/Education/Code/Projets perso - Info/Python_CUPGE1/Projet_fin_semestre/pixels.jpg'


#On vérifie si le chemin existe
if os.path.exists(path):
    img = Image.open(path)
    print(img.size)
    #plt.imshow(img)
    
else :
    print('Chemin introuvable')

#Transformer sous forme de matrice lisible pour l'ordinateur
imgData = np.asarray(img)

#================================================================
# Fonction pour enlever des pixels
#================================================================

def enlever_pixels(image, num_pixels):
    width, height = image.size 


    #Nombres de pixels à choisir aléatoirment afin de les transformer en pixels noirs. 
     

    #On veut retirer des carrées de 4x4 pixels
    square_size = 1

    #On choisit aleatoirement les coordonnées des pixel sà surpprimer
    x_random = np.random.randint(0, width-square_size, num_pixels)
    y_random = np.random.randint(0, height-square_size, num_pixels)

    #Transformer en noir tous les pixels choisit aléatoirement 
    for x, y in zip(x_random, y_random):
        for j in range(square_size):
            for i in range(square_size):
                image.putpixel((x+i, y+j), (0, 255, 0))

    plt.imshow(image)



img2 = img.copy()
img3 = img.copy()

#================================================================
# Méthode 1: La moyenne 
#================================================================

def moyenne_autour(image, patch_size = 6):
    plt.figure(figsize = (5, 3))
    width, height = image.size
    coordonnées_mask = []
    pixels_autour = []
    Moyenne = []
    
    for x in range(width):
        for y in range(height): 
            r, g, b = image.getpixel((x, y))
            if r == 0 and g == 255 and b == 0:
                #Coordonnées des pixels masqué
                coordonnées_mask.append((x, y))
                if x >= patch_size//2 and x < width - patch_size//2 and \
                y >= patch_size//2 and y < height - patch_size//2 :
                    R = 0 
                    G = 0
                    B = 0
                    for i in range(-patch_size//2, patch_size//2 + 1):
                        for j in range(-patch_size//2, patch_size//2 + 1):
                            if i !=0 or j!=0 : 
                                r, g, b = image.getpixel((x+i, y+j))
                                R += r
                                G += g
                                B += b
                                pixels_autour.append((r, g, b))
                                #On met en bleu de patch afin de le visualiser
                                #image.putpixel((x+i, y+j), (0, 0, 255))
                                
                    total_pixels = len(pixels_autour)
                    moy_R = R//total_pixels
                    moy_G = G//total_pixels
                    moy_B = B//total_pixels 

                    Moyenne.append((moy_R, moy_G, moy_B))
                    image.putpixel((x, y), (moy_R, moy_G, moy_B))

    print(coordonnées_mask)
    return(Moyenne)

    #plt.imshow(image)
                             


#================================================================
# Méthode 2: L'écart-type  
#================================================================

def ecart_type(image, patch_size = 1):
    """
    This function computes...
    """
    #plt.figure(figsize = (5, 3))
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            if r == 0 and g == 255 and b == 0:
                if x >= patch_size//2 and x < width - patch_size//2 and \
                y >= patch_size//2 and y < height - patch_size//2 :

                    ecart_type_R = 0
                    ecart_type_G = 0
                    ecart_type_B = 0

                    valeurs_R , valeurs_G, valeurs_B = [], [], []
                    
                    for i in range(-patch_size//2, patch_size//2+1):
                        for j in range(-patch_size//2, patch_size//2+1):
                            if (i, j) != (0, 0):
                                r, g, b = image.getpixel((x+i, y+j))
                                valeurs_R.append(r)
                                valeurs_G.append(g)
                                valeurs_B.append(b)

                    ecart_type_R = np.std(valeurs_R, ddof = 1)
                    ecart_type_G = np.std(valeurs_G, ddof = 1)
                    ecart_type_B = np.std(valeurs_B, ddof = 1)

                    
                    r2 = min(range(256), key=lambda R: abs(np.std(valeurs_R, ddof=1) - ecart_type_R))
                    g2 = min(range(256), key=lambda G: abs(np.std(valeurs_G, ddof=1) - ecart_type_G))
                    b2 = min(range(256), key=lambda B: abs(np.std(valeurs_B, ddof=1) - ecart_type_B))
                    
                    #print((r2, g2, b2))
                    image.putpixel((x, y), (r2, g2, b2))
                print([x, y])
                return((r2, g2, b2))
    #plt.imshow(image)
    
    
#================================================================
# Comparaison   
#================================================================

def erreur_moyenne(image, image2, mask = (0, 255, 0)): 
    width, height = image.size
    total_error = 0
    count = 0 
    for x in range(width):
        for y in range(height):
            if image.getpixel((x, y)) == mask: 
                r, g, b = image.getpixel((x, y))
                r1, g1, b1 = image2.getpixel((x, y))
                total_error += abs(r - r1) + abs(g - g1) + abs(b - b1)
                count += 1
    return total_error/(count*3) if count > 0 else 0

#================================================================
# Comparaison des images  
#================================================================

def afficher_comparaison(image_originale, image_sans_pixel, image_moyenne, image_ecart_type):
    fig, axes = plt.subplots(1, 4, figsize=(12, 9))
    axes[0].imshow(image_originale)
    axes[0].set_title("Image Originale")
    axes[0].axis("off")

    axes[1].imshow(image_sans_pixel)
    axes[1].set_title("Image Pixel manquant")
    axes[1].axis("off")

        # Affichage de la reconstruction
    axes[2].imshow(image_moyenne)
    axes[2].set_title("Prediction (moyenne)")
    axes[2].axis("off")

    axes[3].imshow(image_ecart_type)
    axes[3].set_title("Prediction (ecart-type)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()