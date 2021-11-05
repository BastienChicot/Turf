# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:34:28 2021

@author: basti
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/courses_complet.csv",sep=";")

df.info()

df['Cotes'] = df['Cotes'].str.replace(u"/1Parier", "")
df['date'] = df['date'].str.replace(u"Partants - ", "")
df['Gains'] = df['Gains'].str.replace(u" €", "")
df['Gains'] = df['Gains'].str.replace(u" ", "")

df.loc[df['jockeys'].isna(),'jockeys'] = df['drivers']

df = df[df["S/A"] != "2020.0"]
df = df[df["S/A"] != "2021.0"]

df['S/A'] = df['S/A'].str.replace(u"/", "")
df['S/A'] = df['S/A'].str.replace(u"-", "")

df['sexe'] = df['S/A'].astype(str).str[0]
df['age'] = df['S/A'].astype(str).str[1:]

df[["jour","day","month","year"]] = df["date"].str.split(" ",expand=True)

test=df.copy()
df=test.copy()
df['R/K'] = df['R/K'].str.replace(u"'", ".")
df['R/K'] = df['R/K'].astype(str).str[0:4]

df['gagnant'] = np.where(df['Place']==1, '1', '0')
df["g_place"] = np.where((df["Place"]==1) | (df["Place"]==2) | (df["Place"]==3), "1", "0")
df["quinte"] = np.where((df["Place"]==1)  | (df["Place"]==2) | (df["Place"]==3) |
                        (df["Place"]==4)  | (df["Place"]==5), 1, 0)

df.loc[(df['Corde']>=1) & (df["Corde"]<8),'corde_'] = "int"
df.loc[(df['Corde']>=8) & (df["Corde"]<16),'corde_'] = "milieu"
df.loc[df['Corde']>=16,'corde_'] = "ext"
df.loc[df['corde_'].isna(),'corde_'] = "autre"

df["meme_entr"] = np.where(df["jockeys"]==df["Entraineurs"], "1", "0")

df['R/K'] = df['R/K'].astype(float, errors = 'raise')
df["Gains"] = pd.to_numeric(df['Gains'], errors='coerce').convert_dtypes()
df = df[df["Cotes"] != "-Parier"]
df['Cotes'] = df['Cotes'].astype(float, errors = 'raise')
df=df[df["age"]!="an"]
df['age'] = df['age'].astype(float, errors = 'raise')

df.to_csv("data/base_travail.csv",sep=";")
df=df.rename(columns={"R/K": "RK", "Dist.": "Dist"})
df=pd.read_csv("data/base_travail.csv",sep=";")

top_jock = pd.read_csv("data/top20_jockey.csv",sep=";")

df = df.merge(top_jock,on=["jockeys","spec"],how="outer")
df.loc[df['top_20'].isna(),'top_20'] = 0

import statsmodels.formula.api as smf

df_reg = df.copy()
df_reg=df_reg[["gagnant","g_place","quinte","corde_","Gains","Cotes","spec","sexe","Poids","month",
               "meme_entr","top_20","age"]]
df_reg=df_reg.dropna()

m=df.corr()
test=df.loc[df["spec"].isna()] 
np.unique(df["spec"])
df_reg.info()
reg_lin = smf.ols('quinte ~ np.log(Poids) + np.log(age)*C(sexe) + C(corde_) + np.log(Cotes)',
                  data=df_reg).fit()

reg_lin.summary()

reg = smf.logit('quinte ~ np.log(Poids) + np.log(age) + C(corde_) + np.log(Cotes) + C(sexe)'
              , data=df_reg).fit()

reg.summary()

reg_lin_gagnant = smf.ols('gagnant ~ np.log(Poids) + np.log(age)*C(sexe) + C(corde_) + np.log(Cotes)',
                  data=df_reg).fit()

reg_lin_gagnant.summary()

reg_g = smf.logit('gagnant ~ np.log(Poids) + np.log(age) + C(corde_) + np.log(Cotes) + C(sexe)'
              , data=df_reg).fit()

reg_g.summary()

reg_att_lin = smf.ols('quinte ~ np.log(age)*C(sexe) + np.log(Cotes)',
                  data=df_reg).fit()

reg_att_lin.summary()

reg_atte = smf.logit('quinte ~ np.log(age)*C(sexe) + np.log(Cotes)',
                  data=df_reg).fit()

reg_atte.summary()


df_test=pd.DataFrame({"Poids":[50,60,50,50,50,50,50,50,50,50,50],
                      "age":[6,6,9,6,6,6,6,6,6,6,6],
                      "corde_":["ext","ext","ext","int","milieu","autre","ext","ext","ext","ext","ext"],
                      "Cotes":[7,7,7,7,7,7,3,7,7,7,7],
                      "sexe":["F","F","F","F","F","F","F","M","F","F","F"],
                      "meme_entr":[0,0,0,0,0,0,0,0,1,0,0],
                      "spec":["cros","cros","cros","cros","cros","cros","cros","cros","cros","plat","cros"],
                      "top_20":[0,0,0,0,0,0,0,0,0,0,1]})

from joblib import dump

dump(reg_lin, 'models/g_place1.joblib')
dump(reg, 'models/g_place2.joblib')

dump(reg_att_lin, 'models/quinte_lin.joblib')
dump(reg_atte, 'models/quinte.joblib')

dump(reg_lin_gagnant, 'models/gagnant1.joblib')
dump(reg_g, 'models/gagnant2.joblib')

df_reg['pred']=reg.predict(df_reg)

df_reg['pred_lin']=reg_lin.predict(df_reg)

df_reg['pred_atte']=reg_atte.predict(df_reg)

df_reg['pred_lin_atte']=reg_att_lin.predict(df_reg)

df_reg["err_log"]=df_reg["gagnant"]-df_reg["pred"]
df_reg["err_lin"]=df_reg["gagnant"]-df_reg["pred_lin"]

np.mean(df_reg["err_log"])
np.mean(df_reg["err_lin"])

np.std(df_reg["err_log"])
np.std(df_reg["err_lin"])

np.percentile(df_reg["err_log"],2.5)
np.percentile(df_reg["err_lin"],2.5)
np.percentile(df_reg["err_log"],75)
np.percentile(df_reg["err_lin"],75)

df_3q = df_reg[df_reg["pred_lin_atte"]>0.7]
sum(df_3q["quinte"])/len(df_3q)
np.mean(df_3q["Cotes"])
# REGLE DE DECISION == pred_log > 0.65 et pred_lin > 0.57 ===> 75 % de gagnant placé
# COTE max 3.7 ; moyenne 2.2
# model log == pred > 0.37 et model lin == pred >0.25 =+> 50 % de gagnant
# Cote max 2.7 ; moyenne 1.82

import matplotlib.pyplot as plt

df_test=df_reg[df_reg["Gains"]!=0]
plt.hist(df_reg["age"])
plt.hist(np.log(df_reg["age"]))

###SKLEARN MODELS

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer

df_ml = df.copy()

#CREATION X ET Y
df_ml=df_ml[["Poids","jockeys","Gains","Cotes","spec","sexe","month","g_place","meme_entr",
             "top_20","corde_","age"]]
df_ml = df_ml.dropna() 


y = df_ml[["g_place"]]

#min_max_scaler = MinMaxScaler()
#y_scaled = min_max_scaler.fit_transform(y)
#y = pd.DataFrame(y_scaled)
#y = y[0]

X = df_ml[[
        "Poids",
        "top_20",
        "Gains",
        "Cotes",
        "spec",
        "sexe",
        "month",
        "meme_entr",
        "corde_",
        "age"
        ]]

numeric_data = [
    "age",
    "Poids",
    "Gains",
    "Cotes"
                ]
object_data = [
    'top_20',
    'spec',
    'sexe',
    'month',
    'meme_entr',
    'corde_'
        ]


#PIPELINE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

numeric_pipeline = make_pipeline(PolynomialFeatures(2),PowerTransformer(method="yeo-johnson"),SelectKBest(f_regression,
                                                                                        k=10))
object_pipeline = make_pipeline(OneHotEncoder())

preprocessor = make_column_transformer((numeric_pipeline, numeric_data),
                                       (object_pipeline, object_data))

#MODELE

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

Lin_Reg = make_pipeline(preprocessor, LinearRegression())
MLP = make_pipeline(preprocessor, MLPRegressor())
RFR = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=20))
KNN = make_pipeline(preprocessor, KNeighborsRegressor())
Ridge = make_pipeline(preprocessor,RidgeCV())
SGD = make_pipeline(preprocessor, SGDRegressor())

dict_of_models = {'Linéaire': Lin_Reg,
                  "Neural": MLP,
                  "Ridge": Ridge,
                  "SGD" : SGD,
                  "KNN":KNN,
                  "RFR":RFR,
                 }

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluation (model):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test.values.ravel(), y_pred)
    print(mse)
    print(mae)
    print (model.score(X_test, y_test))
    
for name, model in dict_of_models.items():
    print(name)
    evaluation(model) 