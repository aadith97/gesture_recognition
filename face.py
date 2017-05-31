# if you are on 32 bit os
# import Image
import pickle
# 64 bit with pillow:
from PIL import Image
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import os

import random
label = os.listdir("dataset_image")
dataset=[]
for gesture in label:

    images = os.listdir("dataset_image/"+gesture)

    for image in images:
        img = cv2.imread("dataset_image/"+gesture+"/"+image)
        dataset.append((img,gesture))




random.shuffle(dataset)

X_train=[]
y_train=[]
for  input,person in dataset:
    X_train.append(input)
    y_train.append(label.index(gesture))

Xtrain=np.asarray(X_train)
y_train=np.asarray(y_train)




#u_l=(80/100)*new_image.shape[0]
#u_l=int(u_l)
#data_set_train=(new_image[0:u_l],new_image_class[0:u_l])
#data_set_test=(new_image[u_l:],new_image_class[u_l:])

data_set_train=(Xtrain,y_train)

save_label = open("label1.pickle","wb")
pickle.dump(label, save_label)
save_label.close()
