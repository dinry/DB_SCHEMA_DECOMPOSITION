import time
import pdb
import pandas as pd
import numpy as np
from itertools import groupby

df=pd.DataFrame(np.random.randint(0,10,[3,4]),index=np.arange(0,3),columns=['A','B','C','D'])

print(df.loc[0,'B'])
#访问某个值

"""
访问列
"""
print(df.loc[:,'B'])
#访问B列

print(df.loc[:,['B','C']])
#访问B,C列

print(df.loc[1:,'B'])
#访问index为1往后的所有行的B列

print(df.loc[[0,2],['B','C']])
#访问index为0,2的行的，columns为B，C的列

"""
访问行
"""
print(df.loc[0,:])
#访问index=0的行

print(df.loc[1:,:])
#访问index=1往后的所有行

#iloc()
print(df.iloc[0,0])
#访问0行0列

#访问某列
print(df.iloc[:,0])#第0列
#访问某几列
print(df.iloc[:,[1,2]])#第1列和第2列

#访问某行
print(df.iloc[0,:])#第0行

#定义字典
dic={'name':['Tom','Jone','Mark'],
     'age':[30,18,20],
     'gender':['m','f','m']}
#创建DataFrame
df=pd.DataFrame(dic)
print(df)

#根据年龄这一列，进行排序(升序和降序)
df=df.sort_values(by=['age'])
#默认升序排列 #注意如果不再次赋值，df中的数据仍不会改变
print(df)

df=df.sort_values(by=['age'],ascending=False)
#降序排列
print(df)

#值替换
df['gender']=df['gender'].replace('m','male')#注意也是需要再次赋值。
#如果想改变多个df['gender'].replace(['m','f'],['male','female']) m换成male，f换成female
#df['gender'].replace(['m','f'],'male'])  m和f都换成male
print(df)

#重新排列数据中的列的位置
df=df.ix[:,['name','age','gender']]
#利用ix()访问数据的方法来重新调整列的位置
print(df)
