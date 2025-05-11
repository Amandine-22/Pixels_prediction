import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

##############################################
# NN
##############################################
class NN: 
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate 

        self.b1 = np.zeros((1, self.hidden_size))
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def ReLu(self, x): 
        return np.maximum(0, x)
        
    def ReLu_derivative(self, X): 
        return (X>0).astype(float)  #Si true, retourne 1, sinon, retourne 0
    
    def Sigmoid(self, X): 
        return 1/(1+np.exp(-X))
    
    def Sigmoid_derivative(self, X):
        s = self.Sigmoid(X)
        return s*(1-s)
    
    def foward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.ReLu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.Sigmoid(self.z2)
        return self.a2


    def backward(self, X, y, output): 
        m = X.shape[0] #Returns the number of line of the matrix X

        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m 
        db2 = np.sum(dz2, axis=0, keepdims = True) / m  #Car dz2 est une matrice, on doit donc sommer axe par axe

        dz1 = np.dot(dz2, self.W2.T)*self.ReLu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)/m
        db1 = np.sum(dz1, axis=0, keepdims = True) / m  #Car dz1 est une matrice, on doit donc sommer axe par axe

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1


# X: les données d'entrées 
# y : Les pixels à prédire 
# epochs : le nombre d'itérations d'apprentissage
#batch_size : la taille d'entrainement et de mise à jour des poids 

    def train(self, X, y, epochs, batch_size):
        loss = []
        for epoch in range(epochs): 
            m = X.shape[0]
            epoch_loss = 0

            #Mélanger l'ordre d'entrainement afin de vérifier que le modèle n'apprenne pas par coeur l'ordre 
            i = np.random.permutation(m)
            X_shuffle = X[i]
            y_shuffle = y[i]

            for i in range(0, m, batch_size):
                X_batch = X_shuffle[i:min(i+batch_size, m)] 
                y_batch = y_shuffle[i:min(i+batch_size, m)]

                type(X_batch)

                y_pred = self.foward(X_batch)

                #Calcule du MSE (erreur quadratique moyenne)
                mse = np.mean((y_pred - y_batch)**2)
                epoch_loss += mse* len(X_batch) / m

                self.backward(X_batch, y_batch, y_pred)

            loss.append(epoch_loss)

            if epoch%10 == 0 or epoch == epochs - 1:
                print(f'epoch {epoch}/{epochs} avec une perte de {epoch_loss}')
        return loss
    
    def predict(self, X):
        return self.foward(X)
    

##############################################
# Masquer une partie de l'image 
##############################################

def mask(imgs, mask_size):
    width, height = 32, 32

    all_inputs = []
    all_inputs_fl = []
    all_targets = []
    all_targets_fl = []
    all_x_starts = []
    all_y_starts = []

    for img in imgs: 
        x_start = np.random.randint(width - mask_size)
        y_start = np.random.randint(height - mask_size)

        img = np.array(img)

        normalized_img = img.astype(np.float32) / 255.0

        masked_img = normalized_img.copy()
        target = normalized_img[y_start:y_start+mask_size, x_start:x_start+mask_size]
        masked_img[y_start:y_start+mask_size, x_start:x_start+mask_size] = 0

        all_inputs.append(masked_img)
        all_inputs_fl.append(masked_img.flatten())
        all_targets.append(target)
        all_targets_fl.append(target.flatten())
        all_x_starts.append(int(x_start))
        all_y_starts.append(int(y_start))


    return imgs, all_inputs, all_inputs_fl,  all_targets, all_targets_fl, all_x_starts, all_y_starts

##############################################
# Visualiser l'originale, avec le mask, la target region et enfin la prédite
##############################################

def visualization_mask(imgs, masked, target, predicted, x, y, mask_size):
    for i in range(len(imgs)):
        reconstructed = masked[i].copy()
        pred_region = predicted[i].reshape((mask_size, mask_size, 3))
        y_start = y[i]
        x_start = x[i]
        reconstructed[y_start:y_start+mask_size, x_start:x_start+mask_size] = pred_region

        fig, axes = plt.subplots(1, 5)
        axes[0].imshow(imgs[i])
        axes[0].set_title('l\'originale')
        axes[0].axis('off')

        axes[1].imshow(masked[i])
        axes[1].set_title('Avec le masque')
        axes[1].axis('off')

        axes[2].imshow(target[i])
        axes[2].set_title('la target region')
        axes[2].axis('off')

        axes[3].imshow(pred_region)
        axes[3].set_title('Image prédite')
        axes[3].axis('off')

        axes[4].imshow(reconstructed)
        axes[4].set_title('Image reconstruite')
        axes[4].axis('off')

        plt.tight_layout()
        plt.show()

##############################################
# Choisir la catégorie des images qu'on veut étudier
##############################################

def choose_category(X, y, a):
    X_category = []
    y_category = []
    c = 0 
    for i in range(len(X)):
        if y[i] == a:
            X_category.append(X[i]) #.flatten()
            c+= 1
            y_category.append(c)
    return X_category, y_category

##############################################
# Fonction __name__ == '__main__'
##############################################

if __name__ == "__main__":

    target_size = (32, 32)
    mask_size = 3
    a = 4 #Category

    print('Categorisation des données')
    x_train_cat, y_train_cat = choose_category(x_train, y_train, a)

    print( f'Il y a {len(x_train_cat)} images téléchargées')


    print('préparation des données d\'entrainement, application du masque')

    images, masked, masked_fl, target, target_fl, x, y = mask(x_train_cat, mask_size)

    X_train_mask = np.array(masked_fl)
    y_train_mask = np.array(target_fl)

    print('Préparer les données de test, avec masque')

    X_test_cat, _ = choose_category(x_test, y_test, a)
    max_img = 100
    if len(X_test_cat)>max_img:
        X_test_mask = X_test_cat[:max_img]
    
    images_bis, masked_bis, masked_fl_bis, target_bis, target_fl_bis, x_bis, y_bis = mask(X_test_cat, mask_size)
    
    X_test_mask = np.array(masked_fl_bis)
    y_test_mask = np.array(target_fl_bis)

    input_size = 32 * 32 *3
    output_size = mask_size * mask_size * 3 #car il y a 3 données par pixels (RGB)
    hidden_size = 500 #Nombre de neurones, valeur relativement arbitraire (pas trop peu ni trop sinon ils pourraient surapprendre)

    print(f"Configuration du modèle: input={input_size}, hidden={hidden_size}, output={output_size}")
    model = NN(input_size, hidden_size, output_size, learning_rate=0.001)

    # Split entre entraînement et validation
    val_split = 0.1
    split_idx = int(len(X_train_mask) * (1 - val_split))
    X_train_model, X_val = X_train_mask[:split_idx], X_train_mask[split_idx:]
    y_train_model, y_val = y_train_mask[:split_idx], y_train_mask[split_idx:]

    print("Début de l'entraînement...")

    loss = model.train(X_train_model, y_train_model, epochs = 100, batch_size=32)

    plt.figure(figsize=(10, 6))
    plt.plot(loss)
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.show()

    print('Test du modèle...')

    # Évaluer sur les données de test
    y_pred = model.predict(X_test_mask)
    test_loss = np.mean((y_pred - y_test_mask)**2)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Visualiser quelques exemples de reconstruction
    print("Visualisation des reconstructions...")
    visualization_mask(images_bis[:5], masked_bis[:5], target_bis[:5], y_pred[:5], x_bis[:5], y_bis[:5], mask_size)




