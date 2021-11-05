# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:39:32 2021

@author: basti
"""

## QUINTE pred lineaire = 0.625 == 75%  , 0.41 == 60 % ,0.2 == 50%, 0.85 == 90%, 0.7 == 80%
##/ pred log  0.18 == 50 %, 0.39 == 60% , 0.64 == 75%, 0.72 == 80%, 0.85 == 90 %
import os

os.chdir("C:/Users/basti/OneDrive/Bureau/Turf")

from joblib import load

model_quinte_lin = load('models/quinte_lin.joblib')
model_quinte_log = load('models/quinte.joblib')

atte_lin = load('models/g_place_atte1.joblib') # 0.57 et 0.4 pour 50 %
atte_log = load('models/g_place_atte2.joblib') # 0.65

import pandas as pd
from selenium import webdriver
import time

driver = webdriver.Chrome()

def relance_driver():    
    driver.get("https://www.unibet.fr/turf/race/05-11-2021-R1-C3-vincennes-prix-gallea.html")
    time.sleep(1)

def create_df():
    relance_driver()
    
    time.sleep(2)
    
    output_lst = []    

    for div in driver.find_elements_by_xpath('//*[@id="turfrace-betrunners"]'):
        tds = div.find_elements_by_tag_name('li')
        output_lst = [td.text for td in tds]

    list_df=[]
    for elt in range(0,len(output_lst)):
        temp=output_lst[elt].split('\n')
        tempdf=pd.DataFrame(temp)
        tempdf=tempdf.transpose()
        list_df.append(tempdf)
    
    final=pd.concat(list_df)
    final.columns = final.iloc[0]
    final=final.loc[final[" Chevaux"]!=" Chevaux"]

    for i in list(final):
        if (final[i].str.contains("/")==True).all() :
            sa=final[i]
        if (final[i].str.contains("Kg")==True).all() :
            kg=final[i]

    try:
        final["S/A"]=sa
        final["Poids"]=kg
    except:
        pass
    
    final['S/A'] = final['S/A'].str.replace(u"/", "")
    final['S/A'] = final['S/A'].str.replace(u"-", "")
    
    final['sexe'] = final['S/A'].astype(str).str[0]
    final['age'] = final['S/A'].astype(str).str[1:]
    final["age"]=final["age"].astype(float)
        
    final["Cotes"]=final["Cotes"].fillna(final["Musique"])
    
    final[["cote","dec","reste"]]=final["Cotes"].str.split('.',expand=True)
    final['cote'] = final['cote'].fillna("0")
    final['dec'] = final['dec'].fillna("0")
    final['reste'] = final['reste'].fillna("0")
    final["length"]=final["cote"].str.len()
    
    final['decim']=final["dec"].str[:1]
    final["matin"]=final["cote"]+"."+final["decim"]
    final["debut"]=final["dec"].str[1:]
    final["Cotes"]=final["debut"]+"."+final["reste"]
    
    final=final.set_index(final["N°"])
    final.loc[(final['length']>=3) & (final["reste"]=="0"),'Cotes'] = final["matin"].str[2:]
    
    final.loc[final['Cotes']=="NP.0","Cotes"] = 99
    final.loc[final['Cotes']=="0" ,"Cotes"] = final["matin"]
    final.loc[final['Cotes']=="0.0" ,"Cotes"] = final["matin"]
    final.loc[final['Cotes']==".0" ,"Cotes"] = final["matin"]
    final.loc[final['Cotes']=="0." ,"Cotes"] = final["matin"]
    final["Cotes"]=final["Cotes"].astype(float)
                  
    final=final[[" Chevaux","Jockey","S/A","sexe","age","Entraineur","Cotes"]]
    
    final['pred_gp_log']=atte_log.predict(final)
    final['pred_gp_lin']=atte_lin.predict(final)
    final['pred_qt_log']=model_quinte_lin.predict(final)
    final['pred_qt_lin']=model_quinte_log.predict(final)
        
    return(final)

base=create_df()
base=base.reset_index()
base.to_csv("../App web/data/quinte.csv",sep=";")