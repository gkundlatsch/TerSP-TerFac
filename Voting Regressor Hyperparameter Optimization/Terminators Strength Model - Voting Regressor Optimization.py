# -*- coding: utf-8 -*-
"""
A principal diferença desta versão é a utilização do xgboost
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingRegressor
'''

#Primeiro vamos combinar as tabelas de terminadores sintéticos e naturais, separando 80% para treinamento e 20% para teste

#Agora vamos adicionar novos atributos

df =  pd.read_excel('dados_totais.xlsx')
new_df = pd.read_excel('dados_totais.xlsx')

# Agora vamos criar novos atributos baseados nas caracterísiticas de 'A-tract', 'Hairpin' e 'U-tract'
# Primeiro convertemos todos em strings
new_df['A-tract'] = new_df['A-tract'].astype(str)
new_df['Hairpin'] = new_df['Hairpin'].astype(str)
new_df['U-tract'] = new_df['U-tract'].astype(str)

# Calculamos o tamanho de cada um deles, criando novas colunas.
# No nosso caso A-tract e U-tract são constantes, mas poderiam ser variáveis para outros terminadores
new_df['Tamanho A-tract'] = new_df['A-tract'].apply(len)
new_df['Tamanho Hairpin'] = new_df['Hairpin'].apply(len)
new_df['Tamanho U-tract'] = new_df['U-tract'].apply(len)
new_df['Tamanho Loop'] = new_df['Loop'].apply(len)

# A literatura indica que uma maior presença de "A" no A-tract resulta em maior força. Por isso vamos criar um atributo A% 
# Contamos a ocorrência de "A" em cada A-tract
new_df['Contagem_A_A_tract'] = new_df['A-tract'].apply(lambda x: x.count('A'))

# Calculamos a porcentagem de nucleotídeos A 
new_df['A%_total_A_tract'] = (new_df['Contagem_A_A_tract'] / new_df['Tamanho A-tract'])

new_df['A_tract_7'] = new_df['A-tract'].str[1:]
new_df['Contagem_A_7_A_tract'] = new_df['A_tract_7'].apply(lambda x: x.count('A'))
new_df['A%_7_A_tract'] = (new_df['Contagem_A_7_A_tract'] / 7)

new_df['A_tract_6'] = new_df['A-tract'].str[2:]
new_df['Contagem_A_6_A_tract'] = new_df['A_tract_6'].apply(lambda x: x.count('A'))
new_df['A%_6_A_tract'] = (new_df['Contagem_A_6_A_tract'] / 6)

new_df['A_tract_5'] = new_df['A-tract'].str[3:]
new_df['Contagem_A_5_A_tract'] = new_df['A_tract_5'].apply(lambda x: x.count('A'))
new_df['A%_5_A_tract'] = new_df['Contagem_A_5_A_tract'] / 5

new_df['A_tract_4'] = new_df['A-tract'].str[4:]
new_df['Contagem_A_4_A_tract'] = new_df['A_tract_4'].apply(lambda x: x.count('A'))
new_df['A%_4_A_tract'] = new_df['Contagem_A_4_A_tract'] / 4

new_df['A_tract_3'] = new_df['A-tract'].str[5:]
new_df['Contagem_A_3_A_tract'] = new_df['A_tract_3'].apply(lambda x: x.count('A'))
new_df['A%_3_A_tract'] = new_df['Contagem_A_3_A_tract'] / 3

new_df['A_tract_2'] = new_df['A-tract'].str[6:]
new_df['Contagem_A_2_A_tract'] = new_df['A_tract_2'].apply(lambda x: x.count('A'))
new_df['A%_2_A_tract'] = new_df['Contagem_A_2_A_tract'] / 2

new_df['A_tract_1'] = new_df['A-tract'].str[7:]
new_df['Contagem_A_1_A_tract'] = new_df['A_tract_1'].apply(lambda x: x.count('A'))
new_df['A%_1_A_tract'] = new_df['Contagem_A_1_A_tract'] / 1

#Vamos agora calcular a porcentagem dos demais nucleotídeos
new_df['Contagem_C_A_tract'] = new_df['A-tract'].apply(lambda x: x.count('C'))
new_df['C%_A_tract'] = (new_df['Contagem_C_A_tract'] / new_df['Tamanho A-tract'])
new_df['Contagem_G_A_tract'] = new_df['A-tract'].apply(lambda x: x.count('G'))
new_df['G%_A_tract'] = (new_df['Contagem_G_A_tract'] / new_df['Tamanho A-tract'])
new_df['Contagem_U_A_tract'] = new_df['A-tract'].apply(lambda x: x.count('U'))
new_df['U%_A_tract'] = (new_df['Contagem_U_A_tract'] / new_df['Tamanho A-tract'])

new_df['Contagem_C_7_A_tract'] = new_df['A_tract_7'].apply(lambda x: x.count('C'))
new_df['C%_7_A_tract'] = (new_df['Contagem_C_7_A_tract'] / 7)

new_df['Contagem_C_6_A_tract'] = new_df['A_tract_6'].apply(lambda x: x.count('C'))
new_df['C%_6_A_tract'] = (new_df['Contagem_C_6_A_tract'] / 6)

new_df['Contagem_C_5_A_tract'] = new_df['A_tract_5'].apply(lambda x: x.count('C'))
new_df['C%_5_A_tract'] = new_df['Contagem_C_5_A_tract'] / 5

new_df['Contagem_C_4_A_tract'] = new_df['A_tract_4'].apply(lambda x: x.count('C'))
new_df['C%_4_A_tract'] = new_df['Contagem_C_4_A_tract'] / 4

new_df['Contagem_C_3_A_tract'] = new_df['A_tract_3'].apply(lambda x: x.count('C'))
new_df['C%_3_A_tract'] = new_df['Contagem_C_3_A_tract'] / 3

new_df['Contagem_C_2_A_tract'] = new_df['A_tract_2'].apply(lambda x: x.count('C'))
new_df['C%_2_A_tract'] = new_df['Contagem_C_2_A_tract'] / 2

new_df['Contagem_C_1_A_tract'] = new_df['A_tract_1'].apply(lambda x: x.count('C'))
new_df['C%_1_A_tract'] = new_df['Contagem_C_1_A_tract']

new_df['Contagem_G_7_A_tract'] = new_df['A_tract_7'].apply(lambda x: x.count('G'))
new_df['G%_7_A_tract'] = (new_df['Contagem_G_7_A_tract'] / 7)

new_df['Contagem_G_6_A_tract'] = new_df['A_tract_6'].apply(lambda x: x.count('G'))
new_df['G%_6_A_tract'] = (new_df['Contagem_G_6_A_tract'] / 6)

new_df['Contagem_G_5_A_tract'] = new_df['A_tract_5'].apply(lambda x: x.count('G'))
new_df['G%_5_A_tract'] = new_df['Contagem_G_5_A_tract'] / 5

new_df['Contagem_G_4_A_tract'] = new_df['A_tract_4'].apply(lambda x: x.count('G'))
new_df['G%_4_A_tract'] = new_df['Contagem_G_4_A_tract'] / 4

new_df['Contagem_G_3_A_tract'] = new_df['A_tract_3'].apply(lambda x: x.count('G'))
new_df['G%_3_A_tract'] = new_df['Contagem_G_3_A_tract'] / 3

new_df['Contagem_G_2_A_tract'] = new_df['A_tract_2'].apply(lambda x: x.count('G'))
new_df['G%_2_A_tract'] = new_df['Contagem_G_2_A_tract'] / 2

new_df['Contagem_G_1_A_tract'] = new_df['A_tract_1'].apply(lambda x: x.count('G'))
new_df['G%_1_A_tract'] = new_df['Contagem_G_1_A_tract']

new_df['Contagem_U_7_A_tract'] = new_df['A_tract_7'].apply(lambda x: x.count('U'))
new_df['U%_7_A_tract'] = (new_df['Contagem_U_7_A_tract'] / 7)

new_df['Contagem_U_6_A_tract'] = new_df['A_tract_6'].apply(lambda x: x.count('U'))
new_df['U%_6_A_tract'] = (new_df['Contagem_U_6_A_tract'] / 6)

new_df['Contagem_U_5_A_tract'] = new_df['A_tract_5'].apply(lambda x: x.count('U'))
new_df['U%_5_A_tract'] = new_df['Contagem_U_5_A_tract'] / 5

new_df['Contagem_U_4_A_tract'] = new_df['A_tract_4'].apply(lambda x: x.count('U'))
new_df['U%_4_A_tract'] = new_df['Contagem_U_4_A_tract'] / 4

new_df['Contagem_U_3_A_tract'] = new_df['A_tract_3'].apply(lambda x: x.count('U'))
new_df['U%_3_A_tract'] = new_df['Contagem_U_3_A_tract'] / 3

new_df['Contagem_U_2_A_tract'] = new_df['A_tract_2'].apply(lambda x: x.count('U'))
new_df['U%_2_A_tract'] = new_df['Contagem_U_2_A_tract'] / 2

new_df['Contagem_U_1_A_tract'] = new_df['A_tract_1'].apply(lambda x: x.count('U'))
new_df['U%_1_A_tract'] = new_df['Contagem_U_1_A_tract']

# A literatura indica que a porcentagem de U até o nucleotídeo 7 ou 6 também é importante. Primeiro vamos fazer a contagem total

# Contamos a ocorrência de "U" em cada U-tract
new_df['Contagem_U'] = new_df['U-tract'].apply(lambda x: x.count('U'))

# Calculamos a porcentagem de nucleotídeos U e atribuimos o resultado a uma nova coluna
new_df['U%_Total_U_tract'] = (new_df['Contagem_U'] / new_df['Tamanho U-tract']) 

new_df['U_tract_11'] = new_df['U-tract'].str[:11]
new_df['Contagem_U_tract_11'] = new_df['U_tract_11'].apply(lambda x: x.count('U'))
new_df['U%_11_U_tract'] = new_df['Contagem_U_tract_11'] / 11

new_df['U_tract_10'] = new_df['U-tract'].str[:10]
new_df['Contagem_U_tract_10'] = new_df['U_tract_10'].apply(lambda x: x.count('U'))
new_df['U%_10_U_tract'] = new_df['Contagem_U_tract_10'] / 10

new_df['U_tract_9'] = new_df['U-tract'].str[:9]
new_df['Contagem_U_tract_9'] = new_df['U_tract_9'].apply(lambda x: x.count('U'))
new_df['U%_9_U_tract'] = new_df['Contagem_U_tract_9'] / 9

new_df['U_tract_8'] = new_df['U-tract'].str[:8]
new_df['Contagem_U_tract_8'] = new_df['U_tract_8'].apply(lambda x: x.count('U'))
new_df['U%_8_U_tract'] = new_df['Contagem_U_tract_8'] / 8

new_df['U_tract_7'] = new_df['U-tract'].str[:7]
new_df['Contagem_U_tract_7'] = new_df['U_tract_7'].apply(lambda x: x.count('U'))
new_df['U%_7_U_tract'] = new_df['Contagem_U_tract_7'] / 7

new_df['U_tract_6'] = new_df['U-tract'].str[:6]
new_df['Contagem_U_tract_6'] = new_df['U_tract_6'].apply(lambda x: x.count('U'))
new_df['U%_6_U_tract'] = new_df['Contagem_U_tract_6'] / 6

new_df['U_tract_5'] = new_df['U-tract'].str[:5]
new_df['Contagem_U_tract_5'] = new_df['U_tract_5'].apply(lambda x: x.count('U'))
new_df['U%_5_U_tract'] = new_df['Contagem_U_tract_5'] / 5

new_df['U_tract_4'] = new_df['U-tract'].str[:4]
new_df['Contagem_U_tract_4'] = new_df['U_tract_4'].apply(lambda x: x.count('U'))
new_df['U%_4_U_tract'] = new_df['Contagem_U_tract_4'] / 4

new_df['U_tract_3'] = new_df['U-tract'].str[:3]
new_df['Contagem_U_tract_3'] = new_df['U_tract_3'].apply(lambda x: x.count('U'))
new_df['U%_3_U_tract'] = new_df['Contagem_U_tract_3'] / 3

new_df['U_tract_2'] = new_df['U-tract'].str[:2]
new_df['Contagem_U_tract_2'] = new_df['U_tract_2'].apply(lambda x: x.count('U'))
new_df['U%_2_U_tract'] = new_df['Contagem_U_tract_2'] / 2

new_df['U_tract_1'] = new_df['U-tract'].str[:1]
new_df['Contagem_U_tract_1'] = new_df['U_tract_1'].apply(lambda x: x.count('U'))
new_df['U%_1_U_tract'] = new_df['Contagem_U_tract_1']

new_df['Contagem_C_tract_11'] = new_df['U_tract_11'].apply(lambda x: x.count('C'))
new_df['C%_11_U_tract'] = new_df['Contagem_C_tract_11'] / 11

new_df['Contagem_C_tract_10'] = new_df['U_tract_10'].apply(lambda x: x.count('C'))
new_df['C%_10_U_tract'] = new_df['Contagem_C_tract_10'] / 10

new_df['Contagem_C_tract_9'] = new_df['U_tract_9'].apply(lambda x: x.count('C'))
new_df['C%_9_U_tract'] = new_df['Contagem_C_tract_9'] / 9

new_df['Contagem_C_tract_8'] = new_df['U_tract_8'].apply(lambda x: x.count('C'))
new_df['C%_8_U_tract'] = new_df['Contagem_C_tract_8'] / 8

new_df['Contagem_C_tract_7'] = new_df['U_tract_7'].apply(lambda x: x.count('C'))
new_df['C%_7_U_tract'] = new_df['Contagem_C_tract_7'] / 7

new_df['Contagem_C_tract_6'] = new_df['U_tract_6'].apply(lambda x: x.count('C'))
new_df['C%_6_U_tract'] = new_df['Contagem_C_tract_6'] / 6

new_df['Contagem_C_tract_5'] = new_df['U_tract_5'].apply(lambda x: x.count('C'))
new_df['C%_5_U_tract'] = new_df['Contagem_C_tract_5'] / 5

new_df['Contagem_C_tract_4'] = new_df['U_tract_4'].apply(lambda x: x.count('C'))
new_df['C%_4_U_tract'] = new_df['Contagem_C_tract_4'] / 4

new_df['Contagem_C_tract_3'] = new_df['U_tract_3'].apply(lambda x: x.count('C'))
new_df['C%_3_U_tract'] = new_df['Contagem_C_tract_3'] / 3

new_df['Contagem_C_tract_2'] = new_df['U_tract_2'].apply(lambda x: x.count('C'))
new_df['C%_2_U_tract'] = new_df['Contagem_C_tract_2'] / 2

new_df['Contagem_C_tract_1'] = new_df['U_tract_1'].apply(lambda x: x.count('C'))
new_df['C%_1_U_tract'] = new_df['Contagem_C_tract_1']

new_df['Contagem_G_tract_11'] = new_df['U_tract_11'].apply(lambda x: x.count('G'))
new_df['G%_11_U_tract'] = new_df['Contagem_G_tract_11'] / 11

new_df['Contagem_G_tract_10'] = new_df['U_tract_10'].apply(lambda x: x.count('G'))
new_df['G%_10_U_tract'] = new_df['Contagem_G_tract_10'] / 10

new_df['Contagem_G_tract_9'] = new_df['U_tract_9'].apply(lambda x: x.count('G'))
new_df['G%_9_U_tract'] = new_df['Contagem_G_tract_9'] / 9

new_df['Contagem_G_tract_8'] = new_df['U_tract_8'].apply(lambda x: x.count('G'))
new_df['G%_8_U_tract'] = new_df['Contagem_G_tract_8'] / 8

new_df['Contagem_G_tract_7'] = new_df['U_tract_7'].apply(lambda x: x.count('G'))
new_df['G%_7_U_tract'] = new_df['Contagem_G_tract_7'] / 7

new_df['Contagem_G_tract_6'] = new_df['U_tract_6'].apply(lambda x: x.count('G'))
new_df['G%_6_U_tract'] = new_df['Contagem_G_tract_6'] / 6

new_df['Contagem_G_tract_5'] = new_df['U_tract_5'].apply(lambda x: x.count('G'))
new_df['G%_5_U_tract'] = new_df['Contagem_G_tract_5'] / 5

new_df['Contagem_G_tract_4'] = new_df['U_tract_4'].apply(lambda x: x.count('G'))
new_df['G%_4_U_tract'] = new_df['Contagem_G_tract_4'] / 4

new_df['Contagem_G_tract_3'] = new_df['U_tract_3'].apply(lambda x: x.count('G'))
new_df['G%_3_U_tract'] = new_df['Contagem_G_tract_3'] / 3

new_df['Contagem_G_tract_2'] = new_df['U_tract_2'].apply(lambda x: x.count('G'))
new_df['G%_2_U_tract'] = new_df['Contagem_G_tract_2'] / 2

new_df['Contagem_G_tract_1'] = new_df['U_tract_1'].apply(lambda x: x.count('G'))
new_df['G%_1_U_tract'] = new_df['Contagem_G_tract_1']


new_df['%GC_U_tract'] = new_df['U-tract'].apply(lambda x: x.count('G') + x.count('C'))/12
new_df['%GC_11_U_tract'] = new_df['U_tract_11'].apply(lambda x: x.count('G') + x.count('C'))/11
new_df['%GC_10_U_tract'] = new_df['U_tract_10'].apply(lambda x: x.count('G') + x.count('C'))/10
new_df['%GC_9_U_tract'] = new_df['U_tract_9'].apply(lambda x: x.count('G') + x.count('C'))/9
new_df['%GC_8_U_tract'] = new_df['U_tract_8'].apply(lambda x: x.count('G') + x.count('C'))/8
new_df['%GC_7_U_tract'] = new_df['U_tract_7'].apply(lambda x: x.count('G') + x.count('C'))/7
new_df['%GC_6_U_tract'] = new_df['U_tract_6'].apply(lambda x: x.count('G') + x.count('C'))/6
new_df['%GC_5_U_tract'] = new_df['U_tract_5'].apply(lambda x: x.count('G') + x.count('C'))/5
new_df['%GC_4_U_tract'] = new_df['U_tract_4'].apply(lambda x: x.count('G') + x.count('C'))/4
new_df['%GC_3_U_tract'] = new_df['U_tract_3'].apply(lambda x: x.count('G') + x.count('C'))/3
new_df['%GC_2_U_tract'] = new_df['U_tract_2'].apply(lambda x: x.count('G') + x.count('C'))/2
new_df['%GC_1_U_tract'] = new_df['U_tract_1'].apply(lambda x: x.count('G') + x.count('C'))

new_df['%GC_A_tract'] = new_df['A-tract'].apply(lambda x: x.count('G') + x.count('C'))/8
new_df['%GC_7_A_tract'] = new_df['A_tract_7'].apply(lambda x: x.count('G') + x.count('C'))/7
new_df['%GC_6_A_tract'] = new_df['A_tract_6'].apply(lambda x: x.count('G') + x.count('C'))/6
new_df['%GC_5_A_tract'] = new_df['A_tract_5'].apply(lambda x: x.count('G') + x.count('C'))/5
new_df['%GC_4_A_tract'] = new_df['A_tract_4'].apply(lambda x: x.count('G') + x.count('C'))/4
new_df['%GC_3_A_tract'] = new_df['A_tract_3'].apply(lambda x: x.count('G') + x.count('C'))/3
new_df['%GC_2_A_tract'] = new_df['A_tract_2'].apply(lambda x: x.count('G') + x.count('C'))/2
new_df['%GC_1_A_tract'] = new_df['A_tract_1'].apply(lambda x: x.count('G') + x.count('C'))


new_df['Contagem_A_tract_11'] = new_df['U_tract_11'].apply(lambda x: x.count('A'))
new_df['A%_11_U_tract'] = new_df['Contagem_A_tract_11'] / 11

new_df['Contagem_A_tract_10'] = new_df['U_tract_10'].apply(lambda x: x.count('A'))
new_df['A%_10_U_tract'] = new_df['Contagem_A_tract_10'] / 10

new_df['Contagem_A_tract_9'] = new_df['U_tract_9'].apply(lambda x: x.count('A'))
new_df['A%_9_U_tract'] = new_df['Contagem_A_tract_9'] / 9

new_df['Contagem_A_tract_8'] = new_df['U_tract_8'].apply(lambda x: x.count('A'))
new_df['A%_8_U_tract'] = new_df['Contagem_A_tract_8'] / 8

new_df['Contagem_A_tract_7'] = new_df['U_tract_7'].apply(lambda x: x.count('A'))
new_df['A%_7_U_tract'] = new_df['Contagem_A_tract_7'] / 7

new_df['Contagem_A_tract_6'] = new_df['U_tract_6'].apply(lambda x: x.count('A'))
new_df['A%_6_U_tract'] = new_df['Contagem_A_tract_6'] / 6

new_df['Contagem_A_tract_5'] = new_df['U_tract_5'].apply(lambda x: x.count('A'))
new_df['A%_5_U_tract'] = new_df['Contagem_A_tract_5'] / 5

new_df['Contagem_A_tract_4'] = new_df['U_tract_4'].apply(lambda x: x.count('A'))
new_df['A%_4_U_tract'] = new_df['Contagem_A_tract_4'] / 4

new_df['Contagem_A_tract_3'] = new_df['U_tract_3'].apply(lambda x: x.count('A'))
new_df['A%_3_U_tract'] = new_df['Contagem_A_tract_3'] / 3

new_df['Contagem_A_tract_2'] = new_df['U_tract_2'].apply(lambda x: x.count('A'))
new_df['A%_2_U_tract'] = new_df['Contagem_A_tract_2'] / 2

new_df['Contagem_A_tract_1'] = new_df['U_tract_1'].apply(lambda x: x.count('A'))
new_df['A%_1_U_tract'] = new_df['Contagem_A_tract_1']
    
#Vamos agora calcular a porcentagem dos demais nucleotídeos
new_df['Contagem_C_U_tract'] = new_df['U-tract'].apply(lambda x: x.count('C'))
new_df['C%_U_tract'] = (new_df['Contagem_C_U_tract'] / new_df['Tamanho U-tract'])
new_df['Contagem_G_U_tract'] = new_df['U-tract'].apply(lambda x: x.count('G'))
new_df['G%_U_tract'] = (new_df['Contagem_G_U_tract'] / new_df['Tamanho U-tract'])
new_df['Contagem_A_U_tract'] = new_df['U-tract'].apply(lambda x: x.count('A'))
new_df['A%_U_tract'] = (new_df['Contagem_A_U_tract'] / new_df['Tamanho U-tract'])

# A literatura também indica que o conteúdo GC no início do hairpin é importante.
# Vamos criar o atributo %GC-Base. Primeiro obtemos os 3 primeiros nucleotídeos, depois calculamos conteúdo GC

new_df['Base_Hairpin_3'] = new_df['Hairpin'].str[:3]
new_df['%GC_Base_3'] = new_df['Base_Hairpin_3'].apply(lambda x: x.count('G') + x.count('C'))/3

new_df['Base_Hairpin_2'] = new_df['Hairpin'].str[:2]
new_df['%GC_Base_2'] = new_df['Base_Hairpin_2'].apply(lambda x: x.count('G') + x.count('C'))/2

new_df['Base_Hairpin_1'] = new_df['Hairpin'].str[:1]
new_df['%GC_Base_1'] = new_df['Base_Hairpin_1'].apply(lambda x: x.count('G') + x.count('C'))

# Podemos criar atributos para o hairpin, sem considerar o loop. 
# Para isso vamos primeiro criar uma coluna 'Hairpin sem Loop'
new_df['Hairpin sem Loop'] = ""

# Agora percorremos o dataframe preenchendo a coluna
for index, row in new_df.iterrows():
    hairpin = row['Hairpin']
    loop = row['Loop']
    
    hairpin_sem_loop = hairpin.replace(loop, "")
    
    new_df.at[index, 'Hairpin sem Loop'] = hairpin_sem_loop
    
#Podemos agora calcular quantos nucleotídeos estão pareados no hairpin
new_df['Tamanho Hairpin sem Loop'] = new_df['Hairpin sem Loop'].apply(len)
new_df['N-pareados'] = (new_df['Tamanho Hairpin sem Loop']/2)

#Podemos calcular o conteúdo GC do hairpin sem loop
new_df['%GC_HP'] = new_df['Hairpin sem Loop'].apply(lambda x: x.count('G') + x.count('C'))/new_df['Tamanho Hairpin sem Loop']
new_df['%GC_Loop'] = new_df['Loop'].apply(lambda x: x.count('G') + x.count('C'))/new_df['Tamanho Loop']

new_df['coef_est'] = new_df['Tamanho Hairpin sem Loop'] * new_df['%GC_HP']


new_df['G-posição-6'] = new_df['A-tract'].apply(lambda x: 1 if x[2] == 'G' else 0)
new_df['G-posição-5'] = new_df['A-tract'].apply(lambda x: 1 if x[3] == 'G' else 0)
new_df['G-posição-4'] = new_df['A-tract'].apply(lambda x: 1 if x[4] == 'G' else 0)
new_df['G-posição-3'] = new_df['A-tract'].apply(lambda x: 1 if x[5] == 'G' else 0)
new_df['G-posição-2'] = new_df['A-tract'].apply(lambda x: 1 if x[6] == 'G' else 0)
new_df['G-posição-1'] = new_df['A-tract'].apply(lambda x: 1 if x[7] == 'G' else 0)


new_df['C-posição-6'] = new_df['A-tract'].apply(lambda x: 1 if x[2] == 'C' else 0)
new_df['C-posição-5'] = new_df['A-tract'].apply(lambda x: 1 if x[3] == 'C' else 0)
new_df['C-posição-4'] = new_df['A-tract'].apply(lambda x: 1 if x[4] == 'C' else 0)
new_df['C-posição-3'] = new_df['A-tract'].apply(lambda x: 1 if x[5] == 'C' else 0)
new_df['C-posição-2'] = new_df['A-tract'].apply(lambda x: 1 if x[6] == 'C' else 0)
new_df['C-posição-1'] = new_df['A-tract'].apply(lambda x: 1 if x[7] == 'C' else 0)

#Agora vamos normalizar os atributos que não estão em uma escala de 0 a 1

#Cálculo entropia

new_df['%G_HP'] = new_df['Hairpin sem Loop'].apply(lambda x: x.count('G'))/new_df['Tamanho Hairpin sem Loop']
new_df['%C_HP'] = new_df['Hairpin sem Loop'].apply(lambda x: x.count('C'))/new_df['Tamanho Hairpin sem Loop']
new_df['%A_HP'] = new_df['Hairpin sem Loop'].apply(lambda x: x.count('A'))/new_df['Tamanho Hairpin sem Loop']
new_df['%U_HP'] = new_df['Hairpin sem Loop'].apply(lambda x: x.count('U'))/new_df['Tamanho Hairpin sem Loop']

new_df['%G_Loop'] = new_df['Loop'].apply(lambda x: x.count('G'))/new_df['Tamanho Loop']
new_df['%C_Loop'] = new_df['Loop'].apply(lambda x: x.count('C'))/new_df['Tamanho Loop']
new_df['%A_Loop'] = new_df['Loop'].apply(lambda x: x.count('A'))/new_df['Tamanho Loop']
new_df['%U_Loop'] = new_df['Loop'].apply(lambda x: x.count('U'))/new_df['Tamanho Loop']

def calculate_entropy_A_tract(row):
    probabilities_A_tract = np.array([row['A%_total_A_tract'], row['C%_A_tract'], row['G%_A_tract'], row['U%_A_tract']])
    entropia_A_tract = entropy(probabilities_A_tract, base=2)
    return entropia_A_tract

def calculate_entropy_U_tract(row):
    probabilities_U_tract = np.array([row['U%_Total_U_tract'], row['C%_U_tract'], row['G%_U_tract'], row['A%_U_tract']])
    entropia_U_tract = entropy(probabilities_U_tract, base=2)
    return entropia_U_tract

def calculate_entropy_HP_S_Loop(row):
    probabilities_U_tract = np.array([row['%U_HP'], row['%C_HP'], row['%G_HP'], row['%A_HP']])
    entropia_U_tract = entropy(probabilities_U_tract, base=2)
    return entropia_U_tract

def calculate_entropy_Loop(row):
    probabilities_U_tract = np.array([row['%U_Loop'], row['%C_Loop'], row['%G_Loop'], row['%A_Loop']])
    entropia_U_tract = entropy(probabilities_U_tract, base=2)
    return entropia_U_tract

new_df['Entropia_A_tract'] = new_df.apply(calculate_entropy_A_tract, axis=1)
new_df['Entropia_U_tract'] = new_df.apply(calculate_entropy_U_tract, axis=1)
new_df['Entropia_HP_S_Loop'] = new_df.apply(calculate_entropy_HP_S_Loop, axis=1)
new_df['Entropia_Loop'] = new_df.apply(calculate_entropy_Loop, axis=1)

#Contagem GC Hairpin
def contagem_gc(sequence):
    gc_count = 0
    for char in sequence:
        if char == 'G' or char == 'C':
            gc_count += 1
        elif char == 'A' or char == 'U':
            break
    return gc_count

def calculate_ti_tv_ratio(sequence):
    transitions = ['AG', 'GA', 'CU', 'UC']  # Possible transitions
    transversions = ['AC', 'CA', 'AU', 'UA', 'GC', 'CG', 'GU', 'UG']  # Possible transversions
    
    transition_count = 0
    transversion_count = 0
    
    for i in range(len(sequence) - 1):
        dinucleotide = sequence[i:i+2]
        if dinucleotide in transitions:
            transition_count += 1
        elif dinucleotide in transversions:
            transversion_count += 1
    
    if transversion_count == 0:
        ti_tv_ratio = transition_count
    else:
        ti_tv_ratio = transition_count / transversion_count
    
    return ti_tv_ratio

new_df['Ti_Tv_Ratio_A_Tract'] = new_df['A-tract'].apply(calculate_ti_tv_ratio)
new_df['Ti_Tv_Ratio_Hairpin'] = new_df['Hairpin'].apply(calculate_ti_tv_ratio)
new_df['Ti_Tv_Ratio_U_Tract'] = new_df['U-tract'].apply(calculate_ti_tv_ratio)

def count_state_changes(sequence):
    state_change_count = 0
    previous_nucleotide = sequence[0]

    for nucleotide in sequence[1:]:
        if nucleotide != previous_nucleotide:
            state_change_count += 1
        previous_nucleotide = nucleotide

    return state_change_count

new_df['Hairpin_state-change'] = new_df['Hairpin'].apply(count_state_changes)
new_df['A_Tract_state-change'] = new_df['A-tract'].apply(count_state_changes)
new_df['U_Tract_state-change'] = new_df['U-tract'].apply(count_state_changes)
new_df['HP_S_Loop_state-change'] = new_df['Hairpin sem Loop'].apply(count_state_changes)

# Apply the function to the 'Hairpin' column and create a new column 'GC_before_AU'
new_df['GC_Inicial_Hairpin'] = new_df['Hairpin'].apply(contagem_gc)

#Calculando a e u factor

def calculate_a_factor(sequence):
    a_factor = 0
    for position, nucleotide in enumerate(sequence, start=0):
        if position == 7 and nucleotide == 'A':
            a_factor += 22
        elif position == 6 and nucleotide == 'A':
            a_factor += 16
        elif position == 5 and nucleotide == 'A':
            a_factor += 11
        elif position == 4 and nucleotide == 'A':
            a_factor += 7
        elif position == 3 and nucleotide == 'A':
            a_factor += 3
        elif position == 2 and nucleotide == 'A':
            a_factor += 1
    return a_factor

# Apply the calculate_a_factor function to each sequence in the 'a-tract' column
new_df['a-factor'] = new_df['A-tract'].apply(calculate_a_factor)

def calculate_u_factor(sequence):
    u_factor = 0
    for position, nucleotide in enumerate(sequence, start=0):
        if position == 0 and nucleotide == 'U':
            u_factor += 22
        elif position == 1 and nucleotide == 'U':
            u_factor += 16
        elif position == 2 and nucleotide == 'U':
            u_factor += 11
        elif position == 3 and nucleotide == 'U':
            u_factor += 7
        elif position == 4 and nucleotide == 'U':
            u_factor += 3
        elif position == 5 and nucleotide == 'U':
            u_factor += 1
    return u_factor

# Apply the calculate_u_factor function to each sequence in the 'u-tract' column
new_df['u-factor'] = new_df['U-tract'].apply(calculate_u_factor)

#Normalizar a-factor e u-factor
valores_a_factor = new_df['a-factor'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_a_factor_normalizados = scaler.fit_transform(valores_a_factor)
new_df['a-factor'] = valores_a_factor_normalizados

valores_u_factor = new_df['u-factor'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_u_factor_normalizados = scaler.fit_transform(valores_u_factor)
new_df['u-factor'] = valores_u_factor_normalizados

#Normalizar state-change
valores_Hairpin_state_change = new_df['Hairpin_state-change'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_Hairpin_state_change_normalizados = scaler.fit_transform(valores_Hairpin_state_change)
new_df['Hairpin_state-change'] = valores_Hairpin_state_change_normalizados

valores_A_Tract_state_change = new_df['A_Tract_state-change'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_A_Tract_state_change_normalizados = scaler.fit_transform(valores_A_Tract_state_change)
new_df['A_Tract_state-change'] = valores_A_Tract_state_change_normalizados

valores_U_Tract_state_change = new_df['U_Tract_state-change'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_U_Tract_state_change_normalizados = scaler.fit_transform(valores_U_Tract_state_change)
new_df['U_Tract_state-change'] = valores_U_Tract_state_change_normalizados

#Normalizar GC_Inicial_Hairpin
valores_GC_Inicial_Hairpin = new_df['GC_Inicial_Hairpin'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_GC_Inicial_Hairpin_normalizados = scaler.fit_transform(valores_GC_Inicial_Hairpin)
new_df['GC_Inicial_Hairpin'] = valores_GC_Inicial_Hairpin_normalizados

#Normalizar Tamanho Hairpin
valores_tamanho_hairpin = new_df['Tamanho Hairpin sem Loop'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_tamanho_hairpin_normalizados = scaler.fit_transform(valores_tamanho_hairpin)
new_df['Tamanho Hairpin sem Loop'] = valores_tamanho_hairpin_normalizados

#Normalizar Tamanho Loop
valores_tamanho_loop = new_df['Tamanho Loop'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_tamanho_loop_normalizados = scaler.fit_transform(valores_tamanho_loop)
new_df['Tamanho Loop'] = valores_tamanho_loop_normalizados

#Normalizar N-pareados
valores_n_pareados = new_df['N-pareados'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_n_pareados_normalizados = scaler.fit_transform(valores_n_pareados)
new_df['N-pareados'] = valores_n_pareados_normalizados

#Normalizar Coeficiente Estabilidade
valores_tamanho_coef = new_df['coef_est'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_tamanho_coef_normalizados = scaler.fit_transform(valores_tamanho_coef)
new_df['coef_est'] = valores_tamanho_coef_normalizados

#Normalizar Entropia A-tract
valores_entropia_a_tract = new_df['Entropia_A_tract'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_entropia_a_tract_normalizados = scaler.fit_transform(valores_entropia_a_tract)
new_df['Entropia_A_tract'] = valores_entropia_a_tract_normalizados

#Normalizar Entropia U-tract
valores_entropia_u_tract = new_df['Entropia_U_tract'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_entropia_u_tract_normalizados = scaler.fit_transform(valores_entropia_u_tract)
new_df['Entropia_U_tract'] = valores_entropia_u_tract_normalizados

#Normalizar Entropia HP_S_Loop
valores_entropia_HP_S_Loop = new_df['Entropia_HP_S_Loop'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_entropia_HP_S_Loop_normalizados = scaler.fit_transform(valores_entropia_HP_S_Loop)
new_df['Entropia_HP_S_Loop'] = valores_entropia_HP_S_Loop_normalizados

#Normalizar Entropia Loop
valores_entropia_Loop = new_df['Entropia_Loop'].values.reshape(-1, 1)
scaler = MinMaxScaler()
valores_entropia_Loop_normalizados = scaler.fit_transform(valores_entropia_Loop)
new_df['Entropia_Loop'] = valores_entropia_Loop_normalizados



#Exportamos a nova planilha com as novas colunas

new_df.to_excel('dados_totais_atributos_novos.xlsx', index=False)
'''
new_df = pd.read_excel('dados_totais_atributos_novos.xlsx')

colunas_remover = ['Name', 'Standard Deviation of Strength', 'Loop', 'A-tract', 'Hairpin', 'U-tract', 'Tamanho A-tract',
                   'Tamanho U-tract', 'Contagem_A_A_tract', 'A_tract_7', 'A_tract_6', 'A_tract_5', 'A_tract_4',
                   'A_tract_3', 'A_tract_2', 'A_tract_1', 'Contagem_A_7_A_tract', 'Contagem_A_6_A_tract',
                   'Contagem_A_5_A_tract', 'Contagem_A_4_A_tract', 'Contagem_A_3_A_tract', 'Contagem_A_2_A_tract',
                   'Contagem_A_1_A_tract', 'Contagem_C_7_A_tract', 'Contagem_C_6_A_tract', 'Contagem_C_5_A_tract',
                   'Contagem_C_4_A_tract', 'Contagem_C_3_A_tract', 'Contagem_C_2_A_tract', 'Contagem_C_1_A_tract',
                   'Contagem_G_7_A_tract', 'Contagem_G_6_A_tract', 'Contagem_G_5_A_tract', 'Contagem_G_4_A_tract',
                   'Contagem_G_3_A_tract', 'Contagem_G_2_A_tract', 'Contagem_G_1_A_tract', 'Contagem_U_7_A_tract',
                   'Contagem_U_6_A_tract', 'Contagem_U_5_A_tract', 'Contagem_U_4_A_tract', 'Contagem_U_3_A_tract',
                   'Contagem_U_2_A_tract', 'Contagem_U_1_A_tract', 'Contagem_C_A_tract', 'Contagem_G_A_tract',
                   'Contagem_U_A_tract', 'Contagem_U', 'U_tract_11', 'Contagem_U_tract_11', 'U_tract_10',
                   'Contagem_U_tract_10', 'U_tract_9', 'Contagem_U_tract_9', 'U_tract_8', 'Contagem_U_tract_8',
                   'U_tract_7', 'Contagem_U_tract_7', 'U_tract_6', 'Contagem_U_tract_6', 'U_tract_5',
                   'Contagem_U_tract_5', 'U_tract_4', 'Contagem_U_tract_4', 'U_tract_3', 'Contagem_U_tract_3',
                   'U_tract_2', 'Contagem_U_tract_2', 'U_tract_1', 'Contagem_U_tract_1', 'Contagem_C_U_tract',
                   'Contagem_A_tract_11', 'Contagem_A_tract_10', 'Contagem_A_tract_9', 'Contagem_A_tract_8',
                   'Contagem_A_tract_7', 'Contagem_A_tract_6', 'Contagem_A_tract_5', 'Contagem_A_tract_4',
                   'Contagem_A_tract_3', 'Contagem_A_tract_2', 'Contagem_A_tract_1', 'Contagem_C_tract_11',
                   'Contagem_C_tract_10', 'Contagem_C_tract_9', 'Contagem_C_tract_8',
                   'Contagem_C_tract_7', 'Contagem_C_tract_6', 'Contagem_C_tract_5', 'Contagem_C_tract_4',
                   'Contagem_C_tract_3', 'Contagem_C_tract_2', 'Contagem_C_tract_1', 'Contagem_G_tract_11',
                   'Contagem_G_tract_10', 'Contagem_G_tract_9', 'Contagem_G_tract_8',
                   'Contagem_G_tract_7', 'Contagem_G_tract_6', 'Contagem_G_tract_5', 'Contagem_G_tract_4',
                   'Contagem_G_tract_3', 'Contagem_G_tract_2', 'Contagem_G_tract_1', 'Contagem_C_U_tract',
                   'Contagem_G_U_tract', 'Contagem_A_U_tract', 'Base_Hairpin_3', 'Base_Hairpin_2', 'Base_Hairpin_1',
                   'Hairpin sem Loop', 'N-pareados', 'Tamanho Hairpin',
                   'A%_1_A_tract',
                   'A%_1_U_tract',
                   'A%_2_A_tract',  
                   'A%_2_U_tract',
                   'A%_3_A_tract',
                   'A%_3_U_tract',
                   'A%_4_A_tract',
                   'A%_4_U_tract',
                   'A%_5_A_tract',
                   'A%_5_U_tract',
                   #'A%_6_A_tract', #***
                   #'A%_6_U_tract', #***
                   'A%_7_A_tract',
                   'A%_7_U_tract',
                   'A%_8_U_tract',
                   'A%_9_U_tract',
                   'A%_10_U_tract',
                   'A%_11_U_tract',
                   'A%_total_A_tract',
                   'A%_U_tract',
                   'C%_1_A_tract',
                   'C%_1_U_tract',
                   'C%_2_A_tract',
                   'C%_2_U_tract',
                   'C%_3_A_tract',
                   'C%_3_U_tract',
                   'C%_4_A_tract',
                   'C%_4_U_tract',
                   'C%_5_A_tract', 
                   'C%_5_U_tract',
                   #'C%_6_A_tract', #***
                   'C%_6_U_tract',
                   'C%_7_A_tract',
                   'C%_7_U_tract',
                   'C%_8_U_tract',
                   'C%_9_U_tract',
                   'C%_10_U_tract',
                   'C%_11_U_tract',
                   #'C%_U_tract', #***
                   'C%_A_tract',
                   'G%_1_A_tract',
                   'G%_1_U_tract',
                   'G%_2_A_tract',
                   'G%_2_U_tract',
                   'G%_3_A_tract',
                   'G%_3_U_tract',
                   'G%_4_A_tract',
                   'G%_4_U_tract',
                   'G%_5_A_tract',
                   'G%_5_U_tract',
                   'G%_6_A_tract', 
                   'G%_6_U_tract',
                   'G%_7_A_tract',
                   'G%_7_U_tract',
                   'G%_8_U_tract',
                   'G%_9_U_tract',
                   'G%_10_U_tract',
                   'G%_11_U_tract',
                   'G%_A_tract',
                   'G%_U_tract',
                   #'Tamanho Loop', #***
                   #'Tamanho Hairpin sem Loop', #***
                   'U%_1_A_tract',
                   'U%_1_U_tract',
                   'U%_2_A_tract',
                   'U%_2_U_tract',
                   'U%_3_A_tract',
                   'U%_3_U_tract',
                   'U%_4_A_tract',
                   'U%_4_U_tract',
                   'U%_5_A_tract',
                   'U%_5_U_tract',
                   'U%_6_A_tract',
                   #'U%_6_U_tract', #*** 
                   'U%_7_A_tract',
                   'U%_7_U_tract',
                   'U%_8_U_tract',
                   'U%_9_U_tract',
                   #'U%_10_U_tract', #***
                   'U%_11_U_tract',
                   'U%_A_tract',
                   'U%_Total_U_tract',
                   '%GC_Base_1',
                   '%GC_Base_2',
                   '%GC_Base_3',
                   #'%GC_Loop', #***
                   '%GC_HP',
                   '%A_HP',
                   '%C_HP',
                   '%G_HP',
                   '%U_HP',
                   '%A_Loop',
                   '%C_Loop',
                   '%G_Loop',
                   '%U_Loop',
                   #'Entropia_HP_S_Loop', #***
                   'Entropia_Loop',
                   'coef_est',
                   #'Entropia_A_tract', #***
                   #'Entropia_U_tract', #***
                   'Hairpin_state-change',
                   'Ti_Tv_Ratio_Hairpin',
                   #'A_Tract_state-change', #***
                   #'GC_Inicial_Hairpin', #***
                   'Ti_Tv_Ratio_U_Tract',
                   'Ti_Tv_Ratio_A_Tract',
                   #'U_Tract_state-change', #***
                   #'HP_S_Loop_state-change', #***
                   'a-factor',
                   'u-factor',
                   '%GC_U_tract',
                   '%GC_11_U_tract',
                   '%GC_10_U_tract',
                   '%GC_9_U_tract',
                   '%GC_8_U_tract',
                   '%GC_7_U_tract',
                   '%GC_6_U_tract',
                   '%GC_5_U_tract',
                   '%GC_4_U_tract',
                   '%GC_3_U_tract',
                   '%GC_2_U_tract',
                   '%GC_1_U_tract',
                   '%GC_A_tract',
                   '%GC_7_A_tract',
                   '%GC_6_A_tract',
                   '%GC_5_A_tract',
                   '%GC_4_A_tract',
                   '%GC_3_A_tract',
                   '%GC_2_A_tract',
                   '%GC_1_A_tract',
                   'G-posição-6',
                   'G-posição-5',
                   'G-posição-4',
                   'G-posição-3',
                   'G-posição-2',
                   'G-posição-1',
                   'C-posição-6',
                   'C-posição-5',
                   'C-posição-4',
                   'C-posição-3',
                   'C-posição-2',
                   'C-posição-1',
                   


                   
]

new_df.drop(columns=colunas_remover, inplace=True)

#Agora criamos x e y train

target_column = 'Average Strength'

# Extracting the target variable (y) and the features (X)
X = new_df.drop(columns=[target_column])
y = new_df[target_column]

print(X.columns.tolist())


# ——— Base constructors ———
def make_xgb():
    return XGBRegressor(
        objective='reg:squarederror',
        n_estimators=4000,
        learning_rate=0.001,
        max_depth=6,
        min_child_weight=3,
        subsample=0.5,
        random_state=1996,
    )

def make_mlp():
    return MLPRegressor(
        hidden_layer_sizes=(10,20,10),
        activation='relu',
        solver='adam',
        random_state=1996,
        max_iter=20000,
        tol=1e-4,
        verbose=False
    )

# ——— Define your weight grid ———
weight_pairs = [
    (0,1),
    (1,0),    
    (1, 1),
    (2, 1),
    (1, 2),
    (5, 1),
    (10, 1),
]

# ——— Build the list of (name, constructor) combos ———
model_combos = []
for wx, wm in weight_pairs:
    name = f"Voting_{wx:g}xgb_{wm:g}mlp"
    def make_ensemble(wx=wx, wm=wm):
        return VotingRegressor(
            estimators=[('xgb', make_xgb()), ('mlp', make_mlp())],
            weights=[wx, wm]
        )
    model_combos.append((name, make_ensemble))

# ——— 5-fold CV across all combos ———
kf = KFold(n_splits=5, shuffle=True, random_state=18)
cv_results = []

for name, constructor in model_combos:
    fold_train_r2 = []
    fold_train_mse = []
    fold_test_r2  = []
    fold_test_mse = []

    print(f"\nTesting ensemble: {name}")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = constructor()
        model.fit(X_tr, y_tr)

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)

        fold_train_r2.append(r2_score(y_tr, pred_tr))
        fold_train_mse.append(mean_squared_error(y_tr, pred_tr))
        fold_test_r2.append(r2_score(y_te, pred_te))
        fold_test_mse.append(mean_squared_error(y_te, pred_te))

        print(
            f" Fold {fold_idx}: "
            f"train R²={fold_train_r2[-1]:.4f}, test R²={fold_test_r2[-1]:.4f}"
        )

    cv_results.append({
        'combination':    name,
        'train_r2_mean':  np.mean(fold_train_r2),
        'train_r2_std':   np.std(fold_train_r2, ddof=1),
        'train_mse_mean': np.mean(fold_train_mse),
        'train_mse_std':  np.std(fold_train_mse, ddof=1),
        'test_r2_mean':   np.mean(fold_test_r2),
        'test_r2_std':    np.std(fold_test_r2, ddof=1),
        'test_mse_mean':  np.mean(fold_test_mse),
        'test_mse_std':   np.std(fold_test_mse, ddof=1),
    })

# ——— Summarize & Export ———
summary_df = pd.DataFrame(cv_results)[[
    'combination',
    'train_r2_mean','train_r2_std','train_mse_mean','train_mse_std',
    'test_r2_mean','test_r2_std','test_mse_mean','test_mse_std'
]]
summary_df.to_excel("voting_weight_sweep_f.xlsx", index=False)
print("Results saved to 'voting_weight_sweep.xlsx'")

# Sort like before
summary_df = summary_df.sort_values('test_r2_mean', ascending=False).reset_index(drop=True)

labels   = summary_df['combination']
x        = np.arange(len(labels))
width    = 0.35
means_tr = summary_df['train_r2_mean']
errs_tr  = summary_df['train_r2_std']
means_te = summary_df['test_r2_mean']
errs_te  = summary_df['test_r2_std']

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(
    x - width/2,
    means_tr,
    width,
    yerr=errs_tr,
    capsize=6,
    label='Train R\u00B2',
    color='darkorange',
    edgecolor='black',
    alpha=0.8
)
rects2 = ax.bar(
    x + width/2,
    means_te,
    width,
    yerr=errs_te,
    capsize=6,
    label='Test R\u00B2',
    color='steelblue',
    edgecolor='black',
    alpha=0.8
)

ax.set_xlabel("Combination", fontsize=12)
ax.set_ylabel("Mean R\u00B2 (5-fold)", fontsize=12)
ax.set_title("Train vs. Test R\u00B2 by Model Combination", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')

# Match the same y-limit pattern
ax.set_ylim(
    0.0,
    max(means_tr.max() + errs_tr.max(), means_te.max() + errs_te.max()) + 0.05
)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)


def annotate_bars(rects, vals):
    for r, v in zip(rects, vals):
        ax.annotate(f"{v:.2f}",
                    xy=(r.get_x() + r.get_width()/2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

annotate_bars(rects1, means_tr)
annotate_bars(rects2, means_te)

fig.tight_layout()
plt.savefig("voting_train_vs_test_r2_f.png", dpi=300)
plt.show()


