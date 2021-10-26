import pandas as pd
import numpy as np
import pdb

df1=pd.DataFrame({'A':['Tom','Jone','Marry'],'B':[20,18,19],'C':[1000,3000,2000],'D':[1000,30000,2000]},index=['person1','person2','person3'])
pdb.set_trace()
print(df1)

dict_c=set(df1.columns)

L1=list(dict_c)

L=[None,L1]

l=1
while L[l]

