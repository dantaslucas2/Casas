# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from category_encoders.one_hot import OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split

#Carregar dataframe
dataframe = pd.read_csv("/home/max/Documentos/Aprendizado de máquina/Preco de casas/casas/train.csv")
#dataframe_teste = pd.read_csv("/home/max/Documentos/Aprendizado de máquina/Preco de casas/test.csv")

#plotar Gráficos dos atributos
LotArea = dataframe['LotArea']
preco = dataframe['SalePrice']
plt.plot( LotArea, preco)

#Deletar atributos que não serão utilizados 
del dataframe['Id']

#Binarizar variaveis
#MSZoning
#Street
#
one_hot_enc = OneHotEncoder(cols=['MSZoning'])
#treinoAtributosNews = dataframe['MSZoning']
one_hot_enc.fit_transform(dataframe)

#Codifica variavel ordinalmente
encoder = ce.OrdinalEncoder(cols='GarageFinish', mapping=[{ "col":"GarageFinish", "mapping":{'Na':0, 'Unf':1, 'RFn':2, 'Fin':3}}])
dataframenew = encoder.fit_transform(dataframe)

#separar conjunto de treino e de teste | separar atributos do target
x_treino, x_teste, y_treino, y_teste = train_test_split(dataframe.iloc[:, :-1], dataframe.iloc[:, -1:], test_size=0.3, random_state=10)


#ClassificaÇão
from sklearn import svm
clf = svm.SVC(C=1.0)
clf.fit(x_treino, y_treino)
clf.predict(x_treino)


