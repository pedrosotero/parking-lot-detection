import os
import cv2
from sklearn.cluster import KMeans
import numpy as np

def getFiles(path):
    images = []
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    return images

def readImage(img):
    imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(imagem,(224, 224))

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(len(descriptor_list))])
    for i in range(len(descriptor_list)):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def calcular_pontos_retangulo(ponto1, ponto2):
    x1, y1 = ponto1
    x2, y2 = ponto2
    
    return [y1, y2, x1, x2]

def ponto_central_retangulo(ponto1, ponto2):
    x1, y1 = ponto1
    x2, y2 = ponto2
    
    # Calcula as coordenadas do ponto central
    x_centro = (x1 + x2) // 2
    y_centro = (y1 + y2) // 2
    
    return (x_centro - 45, y_centro)