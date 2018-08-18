from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras import optimizers
import numpy as np 
import pandas as pd 
import seaborn as sb
import tensorflow as tf
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
from resizeimage import resizeimage






data_path_grey = '/home/krishna/Documents/Springer/Image_Processing/Dataset/Train/grey'
data_path_clr = '/home/krishna/Documents/Springer/Image_Processing/Dataset/Train/color'
def load_data(path, height, width, num_channels):
    
    images = []
    labels = []
    # classes = os.listdir(path)
    for file in tqdm(os.listdir(path)):
        #img_file = Image.open(path +'/'+ file).convert('L')
        #img_file = np.array(img_file)
        #shape = img_file.shape
        #img_file = np.matrix.flatten(img_file)
        
        #img_file = np.reshape(img_file, (shape[0], shape[1],1))
        
        img_file = cv2.imread(path + '/' + file)

        #img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        #img_file = np.array(img_file)
        #img_file = resizeimage.resize_contain(img_file, [width, height])

        img_file = cv2.resize(img_file, (width, height))
        img_file = np.array(img_file)
        #img_file = np.reshape(img_file, (width, height,1))
        #img_file = np.matrix.flatten(img_file)
        

        img_arr = np.array(img_file)
        label = np.array(file.split('_')[0])
        images.append(img_arr)
        labels.append(label)
        image_data = np.array(images)
        label_data = np.array(labels)
        

    return image_data, label_data

height = 60
width = 60
num_channels = 1
num_classes = 10
batch_size = 24





# In[5]:


img, labels = load_data(data_path_grey, height, width, num_channels)
img1, labels1 = load_data(data_path_clr, height, width, num_channels)
img1, labels1 = img1[:101], labels1[:101]
X=[]
y=[]
for i in range(len(img)):
    X.append(img[i])


    
    
#img, labels = load_data(data_path_clr, height, width, num_channels)
#print(len(X))



for i in range(len(img1)):
    X.append(img1[i])


    
    
#img, labels = load_data(data_path_clr, height, width, num_channels)
#print(len(X))


for i in range(len(labels)):
    y.append(labels[i])


    
    
#img, labels = load_data(data_path_clr, height, width, num_channels)
#print(len(y))



for i in range(len(labels1)):
    y.append(labels1[i])


    
    
#img, labels = load_data(data_path_clr, height, width, num_channels)
#print(len(y))

X=np.array(X)
y=np.array(y)
label_encoder=LabelEncoder()

#input_classes=['clr','gry']
label_encoder.fit(y)
labels_encoded=label_encoder.transform(y)

print(np.unique(labels_encoded))
print(len(labels_encoded))




img_rows, img_cols, img_channel = 60, 60, 3

base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit(X, labels_encoded, epochs=50, batch_size=12)
