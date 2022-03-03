# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from category_encoders.one_hot import OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
"""
#   Column         Dtype  
---  ------         -----  
 2   MSZoning       object  - One-Hot Encodings
 5   Street         object  - One-Hot Encodings
 6   Alley          object  - One-Hot Encodings
 7   LotShape       object  - OrdinalEncoder
 8   LandContour    object  - OrdinalEncoder V
 9   Utilities      object  - OrdinalEncoder
 10  LotConfig      object  - One-Hot Encodings
 11  LandSlope      object  - OrdinalEncoder
 12  Neighborhood   object  - One-Hot Encodings
 13  Condition1     object  - One-Hot Encodings
 14  Condition2     object  - One-Hot Encodings
 15  BldgType       object  - One-Hot Encodings
 16  HouseStyle     object  - OrdinalEncoder
 21  RoofStyle      object  - One-Hot Encodings/ OrdinalEncoder
 22  RoofMatl       object  - One-Hot Encodings/ OrdinalEncoder
 23  Exterior1st    object  - One-Hot Encodings
 24  Exterior2nd    object  - One-Hot Encodings
 25  MasVnrType     object  - One-Hot Encodings
 27  ExterQual      object  - OrdinalEncoder
 28  ExterCond      object  - OrdinalEncoder
 29  Foundation     object  - One-Hot Encodings
 30  BsmtQual       object  - OrdinalEncoder
 31  BsmtCond       object  - OrdinalEncoder
 32  BsmtExposure   object  - OrdinalEncoder
 33  BsmtFinType1   object  - OrdinalEncoder
 35  BsmtFinType2   object  - OrdinalEncoder
 39  Heating        object  - One-Hot Encodings/ OrdinalEncoder
 40  HeatingQC      object  - OrdinalEncoder
 41  CentralAir     object  - One-Hot Encodings
 42  Electrical     object  - OrdinalEncoder
 53  KitchenQual    object  - OrdinalEncoder
 55  Functional     object  - OrdinalEncoder
 57  FireplaceQu    object  - OrdinalEncoder
 58  GarageType     object  - One-Hot Encodings
 60  GarageFinish   object  - OrdinalEncoder
 63  GarageQual     object  - OrdinalEncoder
 64  GarageCond     object  - OrdinalEncoder
 65  PavedDrive     object  - OrdinalEncoder
 72  PoolQC         object  - OrdinalEncoder
 73  Fence          object  - OrdinalEncoder
 74  MiscFeature    object  - One-Hot Encodings
 78  SaleType       object  - One-Hot Encodings
 79  SaleCondition  object  - One-Hot Encodings
"""


#Carregar dataframe
dataframe = pd.read_csv("/home/max/Documentos/Aprendizado de máquina/Preco de casas/Casas/train.csv")

dataframe.decrib

dataframe.corr()
#dataframe = pd.read_csv("train.csv")
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
del dataframe['BsmtFinSF1']
del dataframe['YearRemodAdd']

#plotar Gráficos dos atributos
LotArea = dataframe['LotArea']
preco = dataframe['GarageCars']
#plt.plot( preco, LotArea)

import seaborn as sns
sns.scatterplot(x="GarageFinish", y="SalePrice", data=dataframe, hue="GarageFinish", size="GarageFinish")
#as1.scatterplot(x="BldgType", y="SalePrice", data=dataframe, hue="BldgType", size="BldgType")



one_hot_enc = OneHotEncoder(cols=['MSZoning'])
#treinoAtributosNews = dataframe['MSZoning']
one_hot_enc.fit_transform(dataframe)


#Codifica variavel ordinalmente
#LandContour OrdinalEncoder
#BldgType - OrdinalEncoder
#RoofStyle - OrdinalEncoder
#ExterQual - OrdinalEncoder
#Foundation - OrdinalEncoder
#BsmtQual -OrdinalEncoder
#Heating - OrdinalEncoder
#HeatingQC- OrdinalEncoder
#Electrical - OrdinalEncoder
#KitchenQual - OrdinalEncoder
#GarageFinish - OrdinalEncoder

landContourEencoder = ce.OrdinalEncoder(cols='LandContour', mapping=[{ "col":"LandContour", "mapping":{'Low':0, 'HLS':1, 'Bnk':2, 'Lvl':3}}]) 
#bldgTypeEncoder = ce.OrdinalEncoder(cols='BldgType', mapping=[{ "col":"BldgType", "mapping":{'Duplx':0, '2FmCon':1, 'TwnhsI':2, 'TwnhsE':3, '1Fam':4}}])
roofStyleEncoder = ce.OrdinalEncoder(cols='RoofStyle', mapping=[{ "col":"RoofStyle", "mapping":{'Shed':0, 'Gambrel':1, 'Mansard':2, 'Flat':3, 'Gable':4, 'Hip':5}}])
exterQualEncoder = ce.OrdinalEncoder(cols='ExterQual', mapping=[{ "col":"ExterQual", "mapping":{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex': 4}}])
foundationEncoder = ce.OrdinalEncoder(cols='Foundation', mapping=[{ "col":"Foundation", "mapping":{'Stone':0, 'Slab':1, 'Wood':2, 'BrkTil':3, 'CBlock':4, 'PConc':5}}])
bsmtQualEncoder = ce.OrdinalEncoder(cols='BsmtQual', mapping=[{ "col":"BsmtQual", "mapping":{'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4,'Ex':5}}])
heatingEncoder = ce.OrdinalEncoder(cols='Heating', mapping=[{ "col":"Heating", "mapping":{'Floor':0, 'Wall':1, 'Grav':2, 'OthW':3, 'GasW':4,'GasA':5}}])
heatingQCEncoder = ce.OrdinalEncoder(cols='HeatingQC', mapping=[{ "col":"HeatingQC", "mapping":{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}}])
electricalEncoder = ce.OrdinalEncoder(cols='Electrical', mapping=[{ "col":"Electrical", "mapping":{'Mix':0, 'FuseP':1, 'FuseF':2, 'FuseA':3, 'SBrkr':4}}])
kitchenQualEncoder = ce.OrdinalEncoder(cols='KitchenQual', mapping=[{ "col":"KitchenQual", "mapping":{'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}}])
garageFinishEncoder = ce.OrdinalEncoder(cols='GarageFinish', mapping=[{ "col":"GarageFinish", "mapping":{'Na':0, 'Unf':1, 'RFn':2, 'Fin':3}}])

newDataframe = landContourEencoder.fit_transform(dataframe)
#newDataframe = condition1Encoder.fit_transform(newDataframe)
newDataframe = bldgTypeEncoder.fit_transform(newDataframe)
newDataframe = roofStyleEncoder.fit_transform(newDataframe)
newDataframe = exterQualEncoder.fit_transform(newDataframe)
newDataframe = foundationEncoder.fit_transform(newDataframe)
newDataframe = bsmtQualEncoder.fit_transform(newDataframe)
newDataframe = heatingEncoder.fit_transform(newDataframe)
newDataframe = heatingQCEncoder.fit_transform(newDataframe)
newDataframe = electricalEncoder.fit_transform(newDataframe)
newDataframe = kitchenQualEncoder.fit_transform(newDataframe)
newDataframe = garageFinishEncoder.fit_transform(newDataframe)
#dataframenew = encoder.fit_transform(dataframe)
#separar conjunto de treino e de teste | separar atributos do target

x_treino, x_teste, y_treino, y_teste = train_test_split(dataframe.iloc[:, :-1], dataframe.iloc[:, -1:], test_size=0.3, random_state=10)

#ClassificaÇão
#from sklearn import svm
#clf = svm.SVC(C=1.0)
#clf.fit(x_treino, y_treino)
#clf.predict(x_treino)


