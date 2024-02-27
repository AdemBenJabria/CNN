# Adem BEN JABRIA 
## Université Côte D'Azur - M2 MIAGE IA2
---

# La détection de la rétinopathie diabétique
### Identifier les signes de rétinopathie diabétique dans les images de l'œil

_Ayant de nombreux membres de ma famille atteints de diabète de type 1, ce sujet me tient particulièrement à cœur. C'est cette raison personnelle qui m'a motivé à choisir ce domaine de recherche._

Dans le monde, **537 millions d'adultes** âgés de 20 à 79 ans vivent actuellement avec un diabète, ce qui représente **10,5%** de la population mondiale dans ce groupe d'âge. Ces chiffres sont alarmants et devraient augmenter pour atteindre **643 millions (11,3 %) d'ici 2030** et **783 millions (12,2%) d'ici 2045**. De plus, **212 millions** de personnes vivant avec le diabète ne sont pas diagnostiquées.

En France, la prévalence du diabète de type 2 diagnostiqué est d'environ **4,5%** de la population générale, tandis que le diabète de type 1 concerne environ **10 %** des personnes diabétiques. Avec près de **3,9 millions** de patients diagnostiqués en 2021, dont 10% de type 1, le diabète représente une **épidémie croissante**. Depuis 20 ans, l’incidence augmente de **4% par an**, et il apparaît de plus en plus tôt, notamment chez les enfants de moins de 5 ans.

![Shéma de l'oeil](images\oeil.png)

La **rétinopathie diabétique (DR)**, complication oculaire du diabète, est une préoccupation majeure en France comme dans le reste du monde. Elle peut conduire à la cécité si elle n'est pas détectée et traitée à temps. Actuellement, la détection de la DR est un processus manuel exigeant, réalisé par des cliniciens formés qui examinent des photographies du fond d'œil. Cette méthode est non seulement longue, mais aussi sujette à des retards qui peuvent entraîner des pertes de suivi et des retards de traitement.

Avec la croissance constante du nombre de personnes atteintes de diabète, il est essentiel de développer des méthodes automatisées pour le dépistage de la DR. Ces méthodes, basées sur la classification d'images, la reconnaissance de formes et l'apprentissage automatique, doivent être suffisamment avancées pour être utilisées efficacement en milieu clinique. L'objectif est de créer des modèles automatisés qui peuvent détecter avec précision la DR, réduisant ainsi le besoin d'une expertise et d'un équipement spécialisés, souvent absents dans les régions les plus touchées par le diabète.




# À propos du jeu de données
## À propos des données

Les images consistent en des scans rétiniens filtrés par un filtre gaussien pour détecter la rétinopathie diabétique. Le jeu de données original est disponible sur _[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)_. Ces images sont redimensionnées en 224x224 pixels afin d'entrainer plus facilement (et plus rapidement) le modèle.

Toutes les images sont déjà sauvegardées dans leurs dossiers respectifs selon la gravité/le stade de la rétinopathie diabétique. Un fichier `train.csv` est fournis dans le zip du jeu de données. 

**0 - No_DR**

**1 - Mild**

**2 - Moderate**

**3 - Severe**

**4 - Proliferate_DR**


# Comprendre les stades de la rétinopathie diabétique
Une glycémie, une pression artérielle et des taux de cholestérol élevés, ainsi qu'une augmentation du poids corporel sont associés au diabète non contrôlé et peuvent endommager les délicats vaisseaux sanguins de la rétine, causant une maladie appelée rétinopathie diabétique. Aux premiers stades de la rétinopathie diabétique, la perte de vision peut être prévenue ou limitée ; mais à mesure que la condition progresse, il devient plus difficile de prévenir la perte de vision.

![Oeil Sain VS Oeil Atteint](/models/images/oeilVS.png)

## Rétinopathie Non Proliférative

La rétinopathie diabétique non proliférative est le stade initial et moins grave de la maladie. En raison des dommages causés par des niveaux élevés de sucre dans le sang, les petits vaisseaux sanguins de la rétine commencent à gonfler et à fuir du liquide et du sang. La fuite provoque un gonflement à l'intérieur de la rétine appelé œdème maculaire, qui est une cause courante de déficience visuelle chez les personnes diabétiques.

À mesure que la rétinopathie non proliférative progresse, les vaisseaux sanguins peuvent se bloquer ou se fermer, et de nouveaux vaisseaux sanguins anormaux peuvent commencer à se former dans les mauvaises parties de la rétine. La vision est généralement non affectée aux stades non prolifératifs, mais il est préférable pour l'ophtalmologue de surveiller ces changements de près tous les quelques mois pour s'assurer qu'ils ne s'aggravent pas au stade “prolifératif”.

## Rétinopathie Diabétique Proliférative

La forme la plus grave de la maladie, la rétinopathie diabétique proliférative, se caractérise par un processus connu sous le nom de néovascularisation. Lorsque les vaisseaux sanguins de la rétine sont bloqués, la rétine est privée de l'oxygène et des nutriments dont elle a besoin pour prospérer. Sentant sa réserve de sang limitée, la rétine réagit en développant de nouveaux vaisseaux sanguins anormaux dans les mauvaises parties de la rétine. Mais les nouveaux vaisseaux sanguins ne fournissent pas un flux sanguin adéquat. Ils sont extrêmement fragiles et, à mesure qu'ils grandissent, peuvent fuir du sang dans le corps vitré. Parfois, cette hémorragie provoque des distorsions visuelles appelées corps flottants ; si l'hémorragie est suffisamment grave, elle peut partiellement ou complètement altérer la vision.

Dans certains cas, la formation de ces nouveaux vaisseaux sanguins anormaux conduit au développement de tissu cicatriciel, ou fibrose, dans la rétine. Le tissu cicatriciel peut faire plisser ou “se pincer” la rétine, ou même la détacher de sa position normale le long de l'arrière de l'œil, ce qui est appelé “décollement rétinien tractive”. Selon la complication, une perte de la vision centrale peut survenir.

Une autre complication à surveiller est la croissance de ces vaisseaux sanguins anormaux bloquant l'angle de drainage, le mécanisme naturel par lequel le fluide sort de l'œil. Si cela se produit, la pression à l'intérieur de l'œil peut augmenter très rapidement, provoquant un glaucome à angle fermé néovasculaire.



![Oeil Sain, léger, modéré, sévère, prolifératif](/models/images/differenceoeil.png)
![Oeil Sain, léger, modéré, sévère, prolifératif en image réel](/models/images/images\01234.png)




```python
# Importation des bibliothèques nécessaires.
from tensorflow import lite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random, os
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') #Pour facilité la lecture et la correction

# Chemin d'accès au fichier CSV contenant les données de diagnostic.
csv_file_path = r'C:\Users\ademb\Documents\diabetic retinopathy_detection\train.csv'

# Lecture du fichier CSV et chargement dans un DataFrame pandas.
df = pd.read_csv(csv_file_path)

# Création de dictionnaires pour mapper les diagnostics à des étiquettes binaires et détaillées.
diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# Application des mappages sur le DataFrame pour créer de nouvelles colonnes avec les étiquettes.
df['binary_type'] = df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
      <th>binary_type</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000c1434d8d7</td>
      <td>2</td>
      <td>DR</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001639a390f0</td>
      <td>4</td>
      <td>DR</td>
      <td>Proliferate_DR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0024cdab0c1e</td>
      <td>1</td>
      <td>DR</td>
      <td>Mild</td>
    </tr>
    <tr>
      <th>3</th>
      <td>002c21358ce6</td>
      <td>0</td>
      <td>No_DR</td>
      <td>No_DR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>005b95c28852</td>
      <td>0</td>
      <td>No_DR</td>
      <td>No_DR</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['type'].value_counts().plot(kind='barh')
```




    <Axes: ylabel='type'>




    
![png](output_4_1.png)
    



```python
df['binary_type'].value_counts().plot(kind='barh')
```




    <Axes: ylabel='binary_type'>




    
![png](output_5_1.png)
    



```python
# Division des données en ensembles d'entraînement, de validation et de test de manière stratifiée.
train_intermediate, val = train_test_split(df, test_size = 0.15, stratify = df['type'])
train, test = train_test_split(train_intermediate, test_size = 0.15 / (1 - 0.15), stratify = train_intermediate['type'])

# Affichage de la distribution des étiquettes dans chaque ensemble de données.
print(train['type'].value_counts(), '\n')
print(test['type'].value_counts(), '\n')
print(val['type'].value_counts(), '\n')
```

    type
    No_DR             1263
    Moderate           699
    Mild               258
    Proliferate_DR     207
    Severe             135
    Name: count, dtype: int64 
    
    type
    No_DR             271
    Moderate          150
    Mild               56
    Proliferate_DR     44
    Severe             29
    Name: count, dtype: int64 
    
    type
    No_DR             271
    Moderate          150
    Mild               56
    Proliferate_DR     44
    Severe             29
    Name: count, dtype: int64 
    
    


```python
# Création des répertoires pour stocker les images triées par ensemble et étiquettes.
base_dir = ''

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Suppression des anciens répertoires et création de nouveaux si nécessaire.
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)
```


```python
# Chemin d'accès au répertoire source des images.
src_dir = r'C:\Users\ademb\Documents\diabetic retinopathy_detection\data\gaussian_filtered_images'

# Copie des images dans les répertoires correspondants selon les ensembles d'entraînement, de validation et de test.
for index, row in train.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(train_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)

for index, row in val.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(val_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
 
for index, row in test.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(test_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
```


```python
# Configuration des chemins vers les dossiers contenant les images pour l'entraînement, la validation et les tests.
train_path = 'train'
val_path = 'validation'
test_path = 'test'

# Configuration du générateur d'images pour l'ensemble d'entraînement.
# ImageDataGenerator est utilisé pour prétraiter les images de manière automatisée.
# 'rescale=1./255' est utilisé pour normaliser les valeurs de pixel des images dans la plage [0, 1].
train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    train_path,       # Chemin vers les images d'entraînement.
    target_size=(224,224),  # Redimensionnement des images à 224x224 pixels.
    shuffle = True    # Mélange aléatoire des images pour éviter tout biais d'ordre lors de l'entraînement.
)

# Configuration du générateur d'images pour l'ensemble de validation avec les mêmes paramètres que pour l'entraînement.
val_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    val_path,         # Chemin vers les images de validation.
    target_size=(224,224),  # Taille cible identique pour assurer la cohérence dans le traitement des images.
    shuffle = True    # Mélange des images pour la validation, utile pour évaluer les performances de manière aléatoire.
)

# Configuration du générateur d'images pour l'ensemble de test.
# 'shuffle = False' est utilisé pour le test pour conserver l'ordre des images, utile lors de l'évaluation des prédictions.
test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    test_path,        # Chemin vers les images de test.
    target_size=(224,224),  # Taille cible pour le redimensionnement des images de test.
    shuffle = False   # Pas de mélange pour l'ensemble de test pour correspondre les prédictions avec leurs vraies étiquettes.
)
```

    Found 2562 images belonging to 2 classes.
    Found 550 images belonging to 2 classes.
    Found 550 images belonging to 2 classes.
    


```python
# Construction du modèle de réseau de neurones convolutifs (CNN) pour la classification.
model = tf.keras.Sequential([
    layers.Conv2D(8, (3,3), padding="valid", input_shape=(224,224,3), activation = 'relu'), # Première couche de convolution avec 8 filtres de taille 3x3, une activation ReLU et un padding 'valid' (sans padding).
    layers.MaxPooling2D(pool_size=(2,2)), # Première couche de MaxPooling pour réduire la dimension spatiale de la sortie.
    layers.BatchNormalization(), # Couche de normalisation par lots pour normaliser les activations de la couche précédente.
    layers.Conv2D(16, (3,3), padding="valid", activation = 'relu'), # Deuxième couche de convolution avec 16 filtres de taille 3x3, une activation ReLU.
    layers.MaxPooling2D(pool_size=(2,2)), # Deuxième couche de MaxPooling.
    layers.BatchNormalization(), # Normalisation par lots.
    layers.Conv2D(32, (4,4), padding="valid", activation = 'relu'), # Troisième couche de convolution avec 32 filtres de taille 4x4, une activation ReLU.
    layers.MaxPooling2D(pool_size=(2,2)), # Troisième couche de MaxPooling.
    layers.BatchNormalization(), # Normalisation par lots.
    layers.Flatten(), # Aplatir les sorties pour les transformer en vecteur avant de les passer à la couche dense.
    layers.Dense(32, activation = 'relu'), # Première couche dense avec 32 unités et une activation ReLU.
    layers.Dropout(0.15), # Couche de Dropout pour réduire le surajustement en désactivant aléatoirement 15% des neurones.
    layers.Dense(2, activation = 'softmax') # Couche de sortie avec activation Softmax pour la classification binaire.
])

# Compilation du modèle.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5), # Utilisation de l'optimiseur Adam avec un faible taux d'apprentissage pour une convergence stable.
    loss=tf.keras.losses.BinaryCrossentropy(), # Utilisation de la fonction de perte BinaryCrossentropy pour un problème de classification binaire.
    metrics=['acc'] # Suivi de la précision ('acc') comme métrique pour évaluer la performance du modèle.
)

# Entraînement du modèle sur les données d'entraînement avec validation sur les données de validation.
history = model.fit(
    train_batches,  # Données d'entraînement.
    epochs=33,      # Nombre d'époques pour l'entraînement.
    validation_data=val_batches  # Données de validation pour évaluer la performance du modèle après chaque époque.
)


# Affichage des graphiques de l'évolution de la précision et de la perte au cours des époques.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

    Epoch 1/33
    81/81 [==============================] - 16s 195ms/step - loss: 0.4087 - acc: 0.8353 - val_loss: 0.8708 - val_acc: 0.5073
    Epoch 2/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.2635 - acc: 0.9126 - val_loss: 1.0595 - val_acc: 0.5073
    Epoch 3/33
    81/81 [==============================] - 5s 57ms/step - loss: 0.2320 - acc: 0.9297 - val_loss: 0.9559 - val_acc: 0.5073
    Epoch 4/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.2192 - acc: 0.9321 - val_loss: 0.5440 - val_acc: 0.6327
    Epoch 5/33
    81/81 [==============================] - 5s 57ms/step - loss: 0.2091 - acc: 0.9356 - val_loss: 0.4189 - val_acc: 0.7273
    Epoch 6/33
    81/81 [==============================] - 5s 57ms/step - loss: 0.1982 - acc: 0.9430 - val_loss: 0.2892 - val_acc: 0.8891
    Epoch 7/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1857 - acc: 0.9426 - val_loss: 0.2089 - val_acc: 0.9255
    Epoch 8/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1798 - acc: 0.9442 - val_loss: 0.1821 - val_acc: 0.9364
    Epoch 9/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1675 - acc: 0.9504 - val_loss: 0.1756 - val_acc: 0.9327
    Epoch 10/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1632 - acc: 0.9493 - val_loss: 0.1736 - val_acc: 0.9364
    Epoch 11/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1581 - acc: 0.9516 - val_loss: 0.1706 - val_acc: 0.9364
    Epoch 12/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1501 - acc: 0.9559 - val_loss: 0.1703 - val_acc: 0.9364
    Epoch 13/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1479 - acc: 0.9500 - val_loss: 0.1684 - val_acc: 0.9382
    Epoch 14/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1506 - acc: 0.9532 - val_loss: 0.1661 - val_acc: 0.9382
    Epoch 15/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1345 - acc: 0.9582 - val_loss: 0.1653 - val_acc: 0.9345
    Epoch 16/33
    81/81 [==============================] - 5s 58ms/step - loss: 0.1324 - acc: 0.9598 - val_loss: 0.1646 - val_acc: 0.9382
    Epoch 17/33
    81/81 [==============================] - 5s 59ms/step - loss: 0.1316 - acc: 0.9610 - val_loss: 0.1611 - val_acc: 0.9382
    Epoch 18/33
    22/81 [=======>......................] - ETA: 2s - loss: 0.1017 - acc: 0.9773


```python
# Sauvegarde du modèle entraîné.
model.save('64x3-CNN.model')
```

    INFO:tensorflow:Assets written to: 64x3-CNN.model\assets
    

    INFO:tensorflow:Assets written to: 64x3-CNN.model\assets
    


```python
# Évaluation du modèle sur l'ensemble de test.
loss, acc = model.evaluate_generator(test_batches, verbose=1)
print("Accuracy: ", acc)
```

    18/18 [==============================] - 3s 182ms/step - loss: 0.1481 - acc: 0.9564
    Accuracy:  0.9563636183738708
    

# Détection de la rétinopathie diabétique


```python
# Importation des bibliothèques pour la prédiction sur une nouvelle image.
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Fonction pour prédire la classe d'une image (présence ou non de rétinopathie diabétique).
def predict_class(path):
    img = cv2.imread(path) # Lire l'image à partir du chemin spécifié.
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertir l'image de BGR (format OpenCV par défaut) en RGB.
    RGBImg = cv2.resize(RGBImg, (224, 224)) # Redimensionner l'image pour correspondre à la taille d'entrée attendue par le modèle (224x224).
    plt.imshow(RGBImg) # Afficher l'image RGB redimensionnée.
    image = np.array(RGBImg) / 255.0 # Normaliser les valeurs des pixels de l'image à une plage de [0, 1].
    new_model = tf.keras.models.load_model("64x3-CNN.model") # Charger le modèle entraîné précédemment sauvegardé.
    predict = new_model.predict(np.array([image])) # Prédire la classe de l'image en utilisant le modèle. L'image est encapsulée dans une liste pour correspondre à la forme d'entrée attendue.
    per = np.argmax(predict, axis=1) # Utiliser argmax pour obtenir l'indice de la classe ayant la plus haute probabilité.
    if per == 1:
        print('No DR') # Si l'indice est 1, écrire 'No DR' (Pas de Rétinopathie Diabétique).
    else:
        print('DR') # Sinon, écrire 'DR' (Rétinopathie Diabétique).

```


```python
predict_class(r'C:\Users\ademb\Documents\diabetic retinopathy_detection\data\gaussian_filtered_images\Mild\0a61bddab956.png')
```

    1/1 [==============================] - 0s 56ms/step
    DR
    


    
![png](output_15_1.png)
    

