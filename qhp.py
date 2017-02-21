# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:23:23 2016

@author: pkillad
"""
import pandas as pd
from petl import fromcsv, look, cut, tocsv  

def age_family(test):
    couple = 0
    #if test['variable'] =='Premium Child':
    if test =='Premium Child':
        #df['age'] = 20
        #df['family_size'] = 1
        age = '20'
        family_size = 1
    elif test.find('Premium Couple') != -1:
        age = test[15:17]
        family_size = 2
        couple = 1
    else:
        #idx = test['variable'].find('Age')
            idx = test.find('Age')
        #df['age'] = test[idx+4: idx+6]
       # idx = test.find('Age')
            age= test[idx+4: idx+6]
            if test.find('Premium Adult') !=-1:
                family_size = 1
                couple = 0
            elif test.find('Couple+1')!=-1 or test.find('Individual+2')!=-1 :
                family_size = 3
                if test.find('Couple+1')!=-1:
                    couple =1
            elif test.find('Couple+2')!=-1 or test.find('Individual+3')!=-1:
                family_size =4
                if test.find('Couple+2')!=-1:
                    couple =1
            elif test.find('Individual+1')!=-1:
                family_size = 2
                couple = 0
            else:
                family_size = 5
    return age,family_size,couple
    #return df['age'],df['family_size']
 
    #Load the 2014 table
#table1 = fromcsv('I:/Parimala/dropbox/Dropbox/Thesis/QHP_2014.csv')
    # Alter the colums
#table2 = cut(table1,table1[0][0:42])
#tocsv(table2, 'I:/Parimala/dropbox/Dropbox/Thesis/new2014.csv') 
data = pd.read_csv('I:/Parimala/dropbox/Dropbox/Thesis/new2014.csv')
data1 = pd.read_csv('I:/Parimala/dropbox/Dropbox/Thesis/new2015.csv') 
data2 = pd.read_csv('I:/Parimala/dropbox/Dropbox/Thesis/new2016.csv') 
df =pd.melt(data, id_vars =['State', 'County', 'Metal Level', 'Plan ID - Standard Component',
       'Plan Type', 'Source','Medical Deductible - individual - standard',
       'Medical Ma1imum Out Of Pocket - individual - standard'])
df1 = pd.melt(data1,id_vars =['State', 'County', 'Metal Level', 'Issuer Name',
       'Plan ID (standard component)', 'Plan Type', 'Source','Medical Deductible-individual-standard',
       'Medical Maximum Out Of Pocket - individual - standard'])
del df1['Issuer Name']
df2 = pd.melt(data2,id_vars =['State Code', 'County Name', 'Metal Level',
       'Plan ID (Standard Component)', 'Plan Type', 'Source','Medical Deductible - Individual - Standard',
       'Medical Maximum Out Of Pocket - Individual - Standard'])
cols = df.columns
# Renaming Columns
df.rename(columns={'Metal Level':'Metal_level','Plan Type':'Plan_Type',
'Plan ID - Standard Component':'Plan_id','Medical Deductible - individual - standard':'deductible',
       'Medical Ma1imum Out Of Pocket - individual - standard':'MOOP'},inplace = True)
df1.rename(columns={'Metal Level':'Metal_level','Plan Type':'Plan_Type',
'Plan ID (standard component)':'Plan_id','Medical Deductible-individual-standard':'deductible',
       'Medical Maximum Out Of Pocket - individual - standard':'MOOP'},inplace = True)
df2.rename(columns={'State Code':'State','County Name':'County',
'Metal Level':'Metal_level','Plan Type':'Plan_Type',
'Plan ID (Standard Component)':'Plan_id','Medical Deductible - Individual - Standard':'deductible',
       'Medical Maximum Out Of Pocket - Individual - Standard':'MOOP'},inplace = True)
#Dealing with Inconsistent Data
df['State']=df.State.str.replace('T1','TX')
df3 = df2[(df2.State =='HI')|(df2.State == 'NM')|(df2.State == 'NV')|(df2.State=='OR')]
#df2 = df2[(df2.State !='HI')|(df2.State != 'NM')|(df2.State != 'NV')|(df2.State!='OR')]
df2 = df2[(df2.State !='HI')]
df2 = df2[(df2.State !='NM')]
df2 = df2[(df2.State !='NV')]
df2 = df2[(df2.State !='OR')]
df['County'] = df.County.astype(str).str.lower()
df1['County'] = df1.County.astype(str).str.lower()
df2['County'] = df2.County.astype(str).str.lower()
df['County'] = df.County.str.replace('1','x').str.replace('st. ','saint ')\
.str.replace('-',' ').str.replace('mc ','mc').str.replace('ste. ','sainte ')\
.str.replace('e. baton','east baton')\
.str.replace('w. baton','west baton')\
.str.replace('menomonee','menominee')\
.str.replace('salem city','salem')\
.str.replace('bristol bay borough','bristol bay')\
.str.replace('bristol city','bristol')\
.str.replace('buena visaint city','buena vista city')\
.str.replace('du page','dupage')\
.str.replace('la moure','lamoure')\
.str.replace('manassus city','manassas park city')\
.str.replace('manassus park city','manassas park city')\
.str.replace('northumberlnd','northumberland')\
.str.replace('radford city','radford')\
.str.replace('northwest artic','northwest arctic')\
.str.replace('poquoson','poquoson city')\
.str.replace('saint john baptist','st john the baptist')\
.str.replace('scott bluff','scotts bluff')\
.str.replace('winchesainte city','winchester city')\
.str.replace('yakutat borough','yakutat')\
.str.replace('wrangell city and borough','wrangell')

#df1['County'] = df1.County.str.replace('bristol bay','bristol bay borough')
df1['County'] = df1.County.str.replace('desoto','de soto')\
.str.replace('dekalb','de kalb')\
.str.replace('dewitt','de witt')\
.str.replace('st joseph','saint joseph')\
.str.replace('la paz','lapaz')\
.str.replace('manassas city','manassas park city')\
.str.replace('miami-dade','miami dade')\
.str.replace('bristol bay bay','bristol bay')
#.str.replace('bristol','bristol bay')\

df2['County'] = df2.County.str.replace('la paz','lapaz')\
.str.replace('manassas city','manassas park city')\
.str.replace('-',' ').str.replace('st joseph','saint joseph')\
.str.replace('dekalb','de kalb')\
.str.replace('dewitt','de witt')\
.str.replace('desoto','de soto')

def dataframe_preprocessing(df):

    
#converting columns to categories
    df['State'] = df.State.astype('category')
    df['County'] = df.County.astype('category')
    df['Metal_level'] = df.Metal_level.astype('category')
    df['Plan_Type'] = df.Plan_Type.astype('category')
    df['Source'] = df.Source.astype('category')
#
#Calculating age from the variable value
    df['age'],df['family_size'],df['couple'] = zip(*df.variable.apply(age_family)) 
    df.age.unique() 
#array([20, '21', '27', '30', '40', '50', '60'] 

#Converting age and value columns to categories
    df['age'] = df.age.astype('category')
    df['value'] = df.value.str.replace('$','').str.replace(' ','').str.replace(',','')
    df['value'] = df.value.astype('float')   
    del df['variable'] 

#Converting Deductible and maximum out of pocket according family size
#indivudual standard is multiplied with family size.  
    df['deductible'] = df.deductible.str.replace('$','').str.replace(' ','')\
    .str.replace(',','').str.replace('NotApplicable','0')
    df['deductible'] = df.deductible.astype('float')
    df['deductible_o'] = df.deductible * df.family_size
    df['MOOP'] = df.MOOP.str.replace('$','').str.replace(' ','').str.replace(',','')
    df['MOOP'] = df.MOOP.astype('float')
    df['MOOP_o'] = df.MOOP * df.family_size
    del df['deductible']
    del df['MOOP']
    return df
     
df = dataframe_preprocessing(df)
df1 =dataframe_preprocessing(df1)
df2 = dataframe_preprocessing(df2)
df['year']= 2014
df1['year'] =2015
df2['year']= 2016
res= pd.concat([df,df1])
res1=pd.concat([res,df2])
res =res.dropna()
res1=res1.dropna()
res.to_csv('I:/Parimala/dropbox/Dropbox/Thesis/final1415.csv',header = True,index = False)
#res.to_csv('C:/Users/pkillad/Dropbox/Thesis/final1415.csv',header = True)