# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:04:05 2021

@author: basti
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
import time

driver = webdriver.Chrome()
driver.get("https://www.france-galop.com/fr/hommes-chevaux/chevaux")
time.sleep(10)
for i in range(200):
    but=driver.find_element_by_class_name("btn.more")
    but.click()
    time.sleep(0.5)
#Liste de chevaux de france-galop
haies = pd.read_csv("data/chevaux_haies.csv",sep=";")
plat = pd.read_csv("data/chevaux_plat.csv",sep=";")
obst = pd.read_csv("data/chevaux_obst.csv",sep=";")
steeple = pd.read_csv("data/chevaux_steeple.csv",sep=";")

haies["spec"]="haie"
plat["spec"]="plat"
obst["spec"]="obst"
steeple["spec"]="step"

chevaux=pd.merge(haies,plat,how="outer")
chevaux=pd.merge(chevaux,obst,how="outer")
chevaux=pd.merge(chevaux,steeple,how="outer")

chevaux = chevaux.rename(columns={"Âge":"Age"})

chevaux=chevaux.drop_duplicates()

chevaux.to_csv("data/liste_chevaux.csv",sep=";")
test = chevaux.loc[chevaux['Cheval']=="GLOIRE DU LUPIN"]

#1 janvier 2016 == 80260
#16 octobre 2021 == 256324

course=[i for i in range(257400,257600)]

temp=[]

for i in tqdm(range(257400,257600)):
    
    try :

        url_part="https://www.turf-fr.com/courses-pmu/partants/r1-prix-"+str(i)
        
        r = requests.get(url_part)
        r_html = r.text
        soup = BeautifulSoup(r_html,'html.parser')
        table=soup.find_all('table')
        tab_data = pd.read_html(str(table[0]),header=0)[0]
        
        compet=soup.find('h1',{"id":"current-page-breadcrumb"})
        compet=compet.get_text()
        
        date=soup.find('span',{"class":"baliseH1Categorie"})
        date=date.get_text()
        
        spec=soup.find("p",{"id":"presentation_course"})
        spec=spec.get_text()
        spec=spec.strip()
        spec=spec[0:4]
        spec=spec.lower()
        
        lieu=soup.find("a",{"id":"bc-partants-programmes"})
        lieu=lieu.get_text()
            
        url_res="https://www.turf-fr.com/courses-pmu/arrivees-rapports/r1-prix-"+str(i)
                
        r_res = requests.get(url_res)
        res_html = r_res.text
        soup_res = BeautifulSoup(res_html,'html.parser')
        table_res=soup_res.find_all('table',{"id":"arrivees_rapports_table"})
        tab_data_res = pd.read_html(str(table_res[0]),header=0)[0]
        
        tab_data_res=tab_data_res[["Place","N°"]]
        
        df=pd.merge(tab_data,tab_data_res,on="N°",how="left")
        df["Place"]=df["Place"].fillna(0)
        
        df = df[df['N°'].notna()]
        df['course']=compet
        df['date']=date
        df['spec']=spec
        df["lieu"]=lieu
        
        temp.append(df)
    
    except:
        pass
    
df_fin=pd.concat(temp)

df_fin.to_csv("data/base_inter3.csv",sep=";")
import numpy as np
run=np.unique(df_fin["course"])    

chevaux = pd.read_csv("data/liste_chevaux.csv",sep=";")
data1=pd.read_csv("data/base_inter1.csv",sep=";")
data2=pd.read_csv("data/base_inter2.csv",sep=";")
data3=pd.read_csv("data/base_inter3.csv",sep=";")

df = pd.concat([data1,data2,data3])

df.to_csv("data/courses_complet.csv",sep=";")

df.head(20)

test=pd.merge(df,bourin,on="Chevaux",how="inner")
