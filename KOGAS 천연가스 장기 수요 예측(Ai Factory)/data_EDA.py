import pandas as pd
import numpy as np
path = 'D:\\ai_data/'
df = pd.read_csv(path+'과제1(산업용).csv')
df2 = pd.read_csv(path+'과제1공급량.csv')
## 결측이 없는 데이터지만 습관적으로 확인필요
# aa = []
# c = df[(df['MONTH']==1)][['RP(상대가격)']].reset_index(drop=True)
# # print(c.loc[1])
# for i in range(len(c)):
#     if i+1 == 25:
#         break
#     else :
#         ab = c.loc[i+1] - c.loc[i]
#         aa.append(ab)

# print(np.mean(aa))

# print(df2)


month_data = df2['MONTH'].unique()
# # Index(['YEAR', ' MONTH', '도시가스(톤)_민수용', '도시가스(톤)_산업용', '도시가스(톤)_총합(민수용+산업용)',
# #        '민수용비중', '산업용비중'],
CIVIL_MEAN=[]
IND_MEAN=[]
for j in month_data:
    civil_month = df2[['도시가스(톤)_민수용']][(df2['MONTH']== j )].reset_index(drop=True)
    ind_month = df2[['도시가스(톤)_산업용']][(df2['MONTH']== j )].reset_index(drop=True)
    # print(civil_month)
    CIVIL = []
    IND = []
    for i in range(len(civil_month)):
        if i+1 == len(civil_month):
            break
        else :
            civil2 = civil_month.loc[i+1] - civil_month.loc[i]
            ind2 = ind_month.loc[i+1] - ind_month.loc[i]
            CIVIL.append(civil2)
            IND.append(ind2)
    CIVIL=np.round(np.mean(CIVIL),0)
    CIVIL_MEAN.append(CIVIL)
    
    IND=np.round(np.mean(IND),0)
    IND_MEAN.append(IND)
    
CIVIL_MEAN = (np.array(CIVIL_MEAN)).reshape(-1,1)
IND_MEAN = (np.array(IND_MEAN)).reshape(-1,1)
# IND_MEAN = IND_MEAN
total = np.concatenate((CIVIL_MEAN,IND_MEAN),1)
print(total)

# print(IND_MEAN)

# for i in range(len(civil)):
#     if i+1 == len(civil):
#         break
#     else :
#         civil2 = civil.loc[i+1] - civil.loc[i]

#         # ab = d.loc[i+1] - d.loc[i]
#         bb.append(civil2)
        
# print(np.mean(bb))


# print(np.mean(bb))


