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

#Variaveis selecionadas
#MSSubClass
#MSZoning - Binarizar
#LotFrontage
#LotArea
#Alley - Binarizar
#LotShape - Binarizar
#LandContour OrdinalEncoder
#LotConfig - Binarizar
#LandSlope-  Binarizar
#Neighborhood-  Binarizar
#Condition1 - OrdinalEncoder
#BldgType - OrdinalEncoder
#HouseStyle - Binarizar
#OverallQual
#OverallCond
#YearBuilt
#RoofStyle - OrdinalEncoder
#Exterior1st - Binarizar
#MasVnrType - Binarizar
#ExterQual - OrdinalEncoder
#Foundation - OrdinalEncoder
#BsmtQual -OrdinalEncoder
#TotalBsmtSF
#Heating - OrdinalEncoder
#HeatingQC- OrdinalEncoder
#CentralAir - Binarizar
#Electrical - OrdinalEncoder
#1stFlrSF
#2ndFlrSF
#GrLivArea
#FullBath
#KitchenQual - OrdinalEncoder
#TotRmsAbvGrd
#Fireplaces 
#GarageFinish - OrdinalEncoder
#GarageCars
#SaleType - Binarizar

#Carregar dataframe
dataframe = pd.read_csv("/home/max/Documentos/Aprendizado de máquina/Preco de casas/Casas/train.csv")
#dataframe_teste = pd.read_csv("/home/max/Documentos/Aprendizado de máquina/Preco de casas/test.csv")

#Deletar atributos que não serão utilizados 
del dataframe['Id']
del dataframe['Street']
del dataframe['Utilities']
del dataframe['Condition2']
del dataframe['RoofMatl']
del dataframe['Exterior2nd']
del dataframe['MasVnrArea']
del dataframe['ExterCond']
del dataframe['BsmtCond']
del dataframe['BsmtExposure']
del dataframe['BsmtFinType1']
del dataframe['BsmtFinType2']
del dataframe['BsmtFinSF2']
del dataframe['BsmtUnfSF']
del dataframe['LowQualFinSF']
del dataframe['BsmtFullBath']
del dataframe['BsmtHalfBath']
del dataframe['HalfBath']
del dataframe['BedroomAbvGr']
del dataframe['KitchenAbvGr']
del dataframe['Functional']
del dataframe['FireplaceQu']
del dataframe['GarageType']
del dataframe['GarageYrBlt']
del dataframe['WoodDeckSF']
del dataframe['OpenPorchSF']
del dataframe['EnclosedPorch']
del dataframe['ScreenPorch']
del dataframe['PoolArea']
del dataframe['PoolQC']
del dataframe['Fence']
del dataframe['MiscFeature']
del dataframe['MiscVal']
del dataframe['MoSold']
del dataframe['YrSold']
del dataframe['SaleCondition']
del dataframe['GarageCond']
del dataframe['GarageQual']
del dataframe['GarageArea']
del dataframe['PavedDrive']
del dataframe['3SsnPorch']


#plotar Gráficos dos atributos
LotArea = dataframe['LotArea']
preco = dataframe['GarageCars']
#plt.plot( preco, LotArea)

import seaborn as sns
sns.scatterplot(x="PavedDrive", y="SalePrice", data=dataframe, hue="PavedDrive", size="PavedDrive")
#as1.scatterplot(x="BldgType", y="SalePrice", data=dataframe, hue="BldgType", size="BldgType")


    
    



#Binarizar variaveis
#MSZoning
#Street
#
#
#
#
#
#
#
#
#
#

one_hot_enc = OneHotEncoder(cols=['MSZoning'])
#treinoAtributosNews = dataframe['MSZoning']
one_hot_enc.fit_transform(dataframe)

#Codifica variavel ordinalmente
#
#
#
#
#
#
#
#
encoder = ce.OrdinalEncoder(cols='GarageFinish', mapping=[{ "col":"GarageFinish", "mapping":{'Na':0, 'Unf':1, 'RFn':2, 'Fin':3}}])
dataframenew = encoder.fit_transform(dataframe)

#separar conjunto de treino e de teste | separar atributos do target
x_treino, x_teste, y_treino, y_teste = train_test_split(dataframe.iloc[:, :-1], dataframe.iloc[:, -1:], test_size=0.3, random_state=10)


#ClassificaÇão
from sklearn import svm
clf = svm.SVC(C=1.0)
clf.fit(x_treino, y_treino)
clf.predict(x_treino)


