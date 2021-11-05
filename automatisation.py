# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:23:54 2021

@author: basti
"""
import os

os.chdir("C:/Users/basti/OneDrive/Bureau/Turf")

import pandas as pd
from joblib import load
from selenium import webdriver
import time

g_place_lin=load("models/g_place1.joblib") # 0.57 et 0.4 pour 50 %
g_place_log=load('models/g_place2.joblib') # 0.65

gag_lin=load('models/gagnant1.joblib') # 0.25 
gag_log=load('models/gagnant2.joblib') # 0.37

atte_lin = load('models/g_place_atte1.joblib')
atte_log = load('models/g_place_atte2.joblib') # SIMILAIRE A GAGNANT PLACE

driver = webdriver.Chrome()

def relance_driver():    
    driver.get("https://www.unibet.fr/turf")
    time.sleep(1)
    driver.find_element_by_xpath("""//*[@id="turfnextraces"]/div/article[1]/div/div/a""").click()

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
    
    if all("Corde" and "Poids") in final.columns:
        
        final=final[[" Chevaux","Jockey","Corde","S/A","sexe","age","Poids","Entraineur","Cotes"]]
        
        final['Poids'] = final['Poids'].str.replace(u"Kg","")
        final["Poids"]=final["Poids"].astype(float)
        
        final["Corde"]=final["Corde"].astype(float)
        
        final.loc[(final['Corde']>=1) & (final["Corde"]<8),'corde_'] = "int"
        final.loc[(final['Corde']>=8) & (final["Corde"]<16),'corde_'] = "milieu"
        final.loc[final['Corde']>=16,'corde_'] = "ext"
        final.loc[final['corde_'].isna(),'corde_'] = "autre"
        
        final['pred_gp_log']=g_place_log.predict(final)
        final['pred_gp_lin']=g_place_lin.predict(final)
        
        final['pred_g_lin']=gag_lin.predict(final)
        final['pred_g_log']=gag_log.predict(final)  
        
    else:                
        final=final[[" Chevaux","Jockey","S/A","sexe","age","Entraineur","Cotes"]]
        
        final['pred_gp_log']=atte_log.predict(final)
        final['pred_gp_lin']=atte_lin.predict(final)
        
    return(final)

def check_course():
    df=create_df()
    
    if len(df)<=8:
        print("Attention, pas assez de chevaux!" + "\n")

    gagnants_places=df.loc[(df["pred_gp_log"]>0.65) & (df["pred_gp_lin"]>0.58)]
    
    gagnants_places=gagnants_places.reset_index()
        
    if gagnants_places.empty==True:
        print("Y a rien !" + "\n" + "\n")
    else:
        print("WOOOOOOOOOOH, y a ca !! " + "\n" )
        print(gagnants_places[["N°"," Chevaux","Cotes","pred_gp_log",'pred_gp_lin']])
        print("\n")

#SI Y A BEAUCOUP D'INEDIT DANS LA COURSE = FREE MONEY AVEC LES CHEVAUX EXPERIMENTES

i=1

while i < 30:
    check_course()
    time.sleep(110)    
    i+=1