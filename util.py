import numpy as np
import cv2
import os
from PIL import Image
import csv
from sklearn.model_selection import train_test_split 
from datasets.HAM import HAM_Dataset
from torch.utils.data import DataLoader

def read_image(path, labels):
    ishape = 224
    images = [file for file in os.listdir(path)]
    x, y = [], []
    count = 0
    for img in images:
        if img == 'ATTRIBUTION.txt' or img == 'LICENSE.txt':
            continue
        print('processing pic' + str(count))
        y.append(labels[img.replace('.jpg', '')])
        img = Image.open(path + '\\' + img)
        img = img.resize((ishape, ishape), Image.ANTIALIAS)
        img1 = np.reshape(img, (3, ishape, ishape))
        x.append(img1)
        count += 1
    x = np.asarray(x) / 255.0
    y = np.asarray(y)
    return x, y

def read_label(path):
    labels = {} # empty dict
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            x = line.split(',')
            x[-1] = x[-1].strip()   # remove \n
            # temp = []
            for i in range(1, 8):
                # temp.append(float(x[i]))    # one-hot coded labels
                if (float(x[i]) == 1.0):
                    labels[x[0]] = i-1
                    break
            # labels[x[0]] = temp
    return labels

def generate_loader(batch_size, test_size = 0.3, shuffle_train = True, shuffle_test = False, pin_memory=True):
    labels = read_label('data\\GroundTruth.csv')
    x, y = read_image('data\\images', labels)
    print(x.shape, y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    train_set = HAM_Dataset(x_train, y_train)
    test_set = HAM_Dataset(x_test, y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train,
                              drop_last=False, pin_memory=pin_memory)
                                  
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle_test,
                              drop_last=False, pin_memory=pin_memory)
                    
    return train_loader, test_loader