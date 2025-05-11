#Import des modules 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import os

#================================================================
# Predictions d'une partie de l'image
#================================================================

path = '/Users/amandine/Desktop/Perso/Education/Code/Projets perso - Info/Python_CUPGE1/Projet_fin_semestre/pixels.jpg'

#plt.figure(figsize = (5, 3))
#On vérifit si le chemin existe
if os.path.exists(path):
    img4 = Image.open(path)
    #print(img.size)
    
else :
    print('Chemin introuvable')


#================================================================
# Fonction pour enlever un carré 
#================================================================ 
    
def carré(image, square_size, num_pixels = 1): 
    width, height = image.size 

    #On choisit aleatoirement les coordonnées des pixel sà surpprimer
    x_random = np.random.randint(0, width-square_size, num_pixels)
    y_random = np.random.randint(0, height-square_size, num_pixels)

    #Transformer en vert tous les pixels choisit aléatoirement 
    for x, y in zip(x_random, y_random):
        for j in range(square_size):
            for i in range(square_size):
                image.putpixel((x+i, y+j), (0, 255, 0))

    dict = {}
    x_coord = []
    y_coord = []
    for x in range(width): 
        for y in range(height): 
            r, g, b = image.getpixel((x, y))
            if (r, g, b) == (0, 255, 0):
                x_coord.append(x)
                y_coord.append(y)
                dict[x] = y

    coin_x = min(x_coord)
    coin_y = dict[coin_x]

    print('Coin supérieur à gauche du carré vert de coordonnées:', coin_x, coin_y)
    print(f'Taille du carré vert {square_size} x {square_size}')
    #img4.putpixel((coin_x, coin_y), (0, 0, 255))
    plt.imshow(image)



#================================================================
# En utilisant la moyenne 
#================================================================

def moyenne_carré(image, mask = (0, 255, 0), square_size = 50):
    width, height = image.size
    dict = {}
    x_coord = []
    y_coord = []
    for x in range(width): 
        for y in range(height): 
            r, g, b = image.getpixel((x, y))
            if (r, g, b) == mask:
                x_coord.append(x)
                y_coord.append(y)
                dict[x] = y
    if len(x_coord) != 0 and len(y_coord) !=0:
        coin_x = min(x_coord)
        coin_y = dict[coin_x]

    else : 
        coin_x, coin_y = 0, 0
    #print(coin_x, coin_y)
    
    if (coin_x, coin_y) != (0, 0): 
        if (coin_y+2) <= height: 
            r1, g1, b1 = image.getpixel((coin_x, coin_y+1))
            #print('couleurs pixel 1:', r1, g1, b1)
            r2, g2, b2 = image.getpixel((coin_x, coin_y+2))
            #print('couleurs pixel 2:', r2, g2, b2)
            moy_r = (r1 + r2)//2
            moy_g = (g1 + g2)//2
            moy_b = (b1 + b2)//2
            #print((coin_x, coin_y), (moy_r, moy_g, moy_b))
            image.putpixel((coin_x, coin_y), (moy_r, moy_g, moy_b))
            #print(image)
            moyenne_carré(image, mask = (0, 255, 0))
            
    #plt.imshow(image)

#================================================================
# En utilisant la moyenne en escargot 
#================================================================

def moyenne_escargot(image, mask=(0, 255, 0), visited=None, step_index=0, x=None, y=None, dx=1, dy=0):
    width, height = image.size
    mask_pixels = [(x, y) for x in range(width) for y in range(height) if image.getpixel((x, y)) == mask]

    if not mask_pixels:
        print("Aucun pixel masqué trouvé.")
        return image

    # Définir les limites du carré à remplir
    x_min = min(x for x, y in mask_pixels)
    y_min = min(y for x, y in mask_pixels)
    x_max = max(x for x, y in mask_pixels)
    y_max = max(y for x, y in mask_pixels)

    if visited is None:
        visited = set()
        x, y = x_min, y_min  # Départ en haut à gauche

    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Droite, Bas, Gauche, Haut

    # Si le pixel est à remplir
    if (x, y) in mask_pixels:
        voisins = [(x + dx, y + dy) for dx, dy in steps]
        voisins_valides = [image.getpixel((vx, vy)) for vx, vy in voisins if (vx, vy) not in mask_pixels and 0 <= vx < width and 0 <= vy < height]
        
        if voisins_valides:
            moy_r = sum(r for r, g, b in voisins_valides) // len(voisins_valides)
            moy_g = sum(g for r, g, b in voisins_valides) // len(voisins_valides)
            moy_b = sum(b for r, g, b in voisins_valides) // len(voisins_valides)
            image.putpixel((x, y), (moy_r, moy_g, moy_b))

    visited.add((x, y))

    # Calculer le prochain pixel
    next_x, next_y = x + dx, y + dy

    # Vérifier si on doit changer de direction
    if next_x < x_min or next_x > x_max or next_y < y_min or next_y > y_max or (next_x, next_y) in visited:
        step_index = (step_index + 1) % 4  # Passer à la prochaine direction en Modulo 4
        dx, dy = steps[step_index]
        next_x, next_y = x + dx, y + dy

    # Condition d'arrêt
    if len(visited) == len(mask_pixels):
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        return image

    # Appel récursif
    return moyenne_escargot(image, mask, visited, step_index, next_x, next_y, dx, dy)


#================================================================
# En utilisant l'écart-type en escargot
#================================================================

def ecart_type_escargot(image, mask=(0, 255, 0), visited=None, step_index=0, x=None, y=None, dx=1, dy=0):
    width, height = image.size
    mask_pixels = [(x, y) for x in range(width) for y in range(height) if image.getpixel((x, y)) == mask]

    if not mask_pixels:
        print("Aucun pixel masqué trouvé.")
        return image

    # Définir les limites du carré à remplir
    x_min = min(x for x, y in mask_pixels)
    y_min = min(y for x, y in mask_pixels)
    x_max = max(x for x, y in mask_pixels)
    y_max = max(y for x, y in mask_pixels)

    if visited is None:
        visited = set()
        x, y = x_min, y_min  # Départ en haut à gauche

    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Droite, Bas, Gauche, Haut

    # Si le pixel est à remplir
    if (x, y) in mask_pixels:
        voisins = [(x + dx, y + dy) for dx, dy in steps]
        voisins_valides = [image.getpixel((vx, vy)) for vx, vy in voisins if (vx, vy) not in mask_pixels and 0 <= vx < width and 0 <= vy < height]
        sumR, sumG, sumB = [], [], []
        if voisins_valides:
            sumR.extend([r for r, g, b in voisins_valides])
            sumG.extend([g for r, g, b in voisins_valides])
            sumB.extend([b for r, g, b in voisins_valides])
            
            if len(sumR) > 1:
                E_T_R = int(np.std(sumR, ddof = 1))
                E_T_G = int(np.std(sumG, ddof = 1))
                E_T_B = int(np.std(sumB, ddof = 1))
            
            else:
                E_T_R = 0
                E_T_G = 0
                E_T_B = 0

            image.putpixel((x, y), (E_T_R, E_T_G, E_T_B))

    visited.add((x, y))

    # Calculer le prochain pixel
    next_x, next_y = x + dx, y + dy

    # Vérifier si on doit changer de direction
    if next_x < x_min or next_x > x_max or next_y < y_min or next_y > y_max or (next_x, next_y) in visited:
        step_index = (step_index + 1) % 4  # Passer à la prochaine direction en Modulo 4
        dx, dy = steps[step_index]
        next_x, next_y = x + dx, y + dy

    # Condition d'arrêt
    if len(visited) == len(mask_pixels):
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        return image

    # Appel récursif
    return ecart_type_escargot(image, mask, visited, step_index, next_x, next_y, dx, dy)

################################################
# Afficher la comparaison 
################################################

def afficher_comparaison_carrés(image_originale, image_sans_pixel, image_moyenne, image_ecart_type):
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