```python
import numpy as np
import pandas as pd
```

```python
df = pd.read_csv('./data/titanic.csv')
df.head()
```

```python
df.head(10)
```

```python
df.tail()
```

```python
help(pd.read_csv)
```

```python
df.info()
```

```python
df.index
```

```python
df.columns
```

```python
df.dtypes
```

```python
df.values
```

```python
data = {'country':['aaa','bbb','ccc'],'population':[10,12,14]}
df_data = pd.DataFrame(data)
df_data
```

```python
df_data.info()
```

```python
age = df['Age']
age[:5]
```

```python
age.index
```

```python
age.values[:5]
```

```python
df.head()
```

```python
df = df.set_index('Name')
df.head()
```

```python
age = df['Age']
age[:5]
```

```python
age['Braund, Mr. Owen Harris']
```

```python
age = age+10
age[:5]
```

```python
age.mean()
```

```python
age.max()
```

```python
age.min()
```

```python
df.describe()
```

```python
df = pd.read_csv('./data/titanic.csv')
```

```python
df['Age'][:5]
```

```python
df[['Age','Fare']][:5]
```

```python
df.iloc[0]
```

```python
df.iloc[0:5,1:3]
```

```python
df = df.set_index('Name')
df
```

```python
df.loc['Heikkinen, Miss. Laina']
```

```python
df.loc['Heikkinen, Miss. Laina','Fare']
```

```python
df['Fare'] > 40
```

```python
df[df['Fare'] > 40][:5]
```

```python
df[df['Sex'] == 'male'][:5]
```

```python
df.loc[df['Sex'] == 'male', 'Age'].mean()
```

```python
df = pd.DataFrame({'key': ['A','B',"C",'A','B','C','A','B','C'],'data':[0,5,10,5,10,15,10,15,20]})
df
```

```python
for key in ['A','B','C']:
    print(key, df[df['key'] == key].sum())
```

```python
df.groupby('key').sum()
```

```python
df.groupby('key').aggregate(np.mean)
```

```python
df = pd.read_csv('./data/titanic.csv')
```

```python
df.groupby('Sex')['Age'].mean()
```

```python
df.groupby('Sex')['Survived'].mean()
```

```python
df = pd.DataFrame([[1,2,3],[4,5,6]],index=['a','b'],columns=['A','B','C'])
df
```

```python
df.sum()
```

```python
df.sum(axis=0)
```

```python
df.sum(axis=1)
```

```python
df.sum(axis='columns')
```

```python
df.mean()
```

```python
df.mean(axis=1)
```

```python
df.min()
```

```python
df.max()
```

```python
df.median()
```

```python
df = pd.read_csv('./data/titanic.csv')
df.head()
```

```python
df.cov()
```

```python
df.corr()
```

```python
df['Age'].value_counts()
```

```python
df['Age'].value_counts(ascending=True)
```

```python
df['Pclass'].value_counts(ascending=True)
```

```python
df['Age'].value_counts(ascending=True,bins=5)
```

```python
data = [10,11,12]
index = ['a','b','c']
s = pd.Series(data=data,index=index)
s
```

```python
s[0]
```

```python
s[0:2]
```

```python
mask = [True,False,True]
s[mask]
```

```python
s.loc['b']
```

```python
s.iloc[1]
```

```python
s1 = s.copy()
s1['a'] = 100
s1
```

```python
s1.replace(to_replace=100,value=101,inplace=True)
s1
```

```python
s1.index
```

```python
s1.index = ['a', 'b', 'd']
s1
```

```python
s1.rename(index={'a':'A'},inplace=True)
s1
```

```python
data = [100,110]
index = ['h','k']
s2 = pd.Series(data=data,index=index)
s2
```

```python
s3=s1.append(s2)
s3
```

```python
s1['j'] = 50
s1
```

```python
del s1['A']
s1
```

```python
s1.drop(['b','d'], inplace=True)
s1
```

```python
data = [[1,2,3],[4,5,6]]
index = ['a','b']
columns = ['A','B','C']

df = pd.DataFrame(data=data,index=index,columns=columns)
df
```

```python
df['A']
```

```python
df.iloc[0]
```

```python
df.loc['a']
```

```python
df.loc['a']['A']
```

```python
df.loc['a']['A'] = 150
df
```

```python
df.index = ['f','g']
df
```

```python
df.loc['c'] = [1,2,3]
df
```

```python
data = [[1,2,3],[4,5,6]]
index = ['j','k']
columns = ['A','B','C']
df2 = pd.DataFrame(data=data,index=index,columns=columns)
df2
```

```python
df3 = pd.concat([df,df2],axis=0)
df3
```

```python
df2['Test'] = [10,11]
df2
```

```python
df4 = pd.DataFrame([[10,11],[12,13]],index=['j','k'],columns=['D','E'])
df4
```

```python
df5 = pd.concat([df2,df4],axis=1)
df5
```

```python
df5.drop(['j'],axis=0,inplace=True)
df5
```

```python
del df5['Test']
df5
```

```python
df5.drop(['A','B','C'],axis=1,inplace=True)
df5
```

```python
left = pd.DataFrame({'key':['K0','K1','K2','K3'],'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3']})
right= pd.DataFrame({'key':['K0','K1','K2','K3'],'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']})
```

```python
left
```

```python
right
```

```python
res = pd.merge(left,right)
res
```

```python
res = pd.merge(left, right, on='key')
res
```

```python
left = pd.DataFrame({'key1':['K0','K1','K2','K3'],'key2':['K0','K1','K2','K3'],'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3']})
right= pd.DataFrame({'key1':['K0','K1','K2','K3'],'key2':['K0','K1','K2','K4'],'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']})
```

```python
print(left)
print(right)
```

```python
res = pd.merge(left,right, on=['key1','key2'])
res
```

```python
res = pd.merge(left,right, on=['key1','key2'], how='outer')
res
```

```python
res = pd.merge(left,right, on=['key1','key2'], how='outer', indicator=True)
res
```

```python
res = pd.merge(left,right,how='left')
res
```

```python
res = pd.merge(left,right,how='right')
res
```

```python
pd.get_option('display.max_rows')
```

```python
pd.Series(index=range(0,100))
```

```python
pd.set_option('display.max_rows', 6)
```

```python
pd.Series(index=range(0,100))
```

```python
pd.get_option('display.max_columns')
```

```python
pd.DataFrame(columns=range(0,30))
```

```python
pd.set_option('display.max_columns', 20)
```

```python
pd.DataFrame(columns=range(0,30))
```

```python
pd.get_option('display.max_colwidth')
```

```python
pd.Series(index=['A'],data=['t'*70])
```

```python
pd.set_option('display.max_colwidth', 100)
```

```python
pd.Series(index=['A'],data=['t'*70])
```

```python
pd.get_option('display.precision')
```

```python
pd.Series(data=[1.2345678923456])
```

```python
pd.set_option('display.precision',5)
pd.Series(data=[1.2345678923456])
```

```python
example = pd.DataFrame({'Month':["January","January","January","January",
                               "February","February","February","February",
                               "March","March","March","March"],
                       'Category':["Transportation","Grocery","Household","Entertainment",
                                  "Transportation","Grocery","Household","Entertainment",
                                  "Transportation","Grocery","Household","Entertainment"],
                       'Amount':[74.,235.,175.,100.,
                                115.,240.,225.,125.,
                                90.,260.,200.,120.]})
```

```python
example
```

```python
example_pivot = example.pivot(index='Category',columns='Month',values='Amount')
example_pivot
```

```python
example_pivot.sum(axis=1)
```

```python
example_pivot.sum(axis=0)
```

```python
df = pd.read_csv('./data/titanic.csv')
df.head()
```

```python
df.pivot_table(index='Sex',columns='Pclass',values='Fare')
```

```python
df.pivot_table(index='Sex',columns='Pclass',values='Fare',aggfunc='max')
```

```python
df.pivot_table(index='Sex',columns='Pclass',values='Fare',aggfunc='count')
```

```python
df.pivot_table(index='Sex',columns='Pclass',values='Fare',aggfunc='mean')
```

```python
df['Underaged'] = df['Age'] <= 18
```

```python
df
```

```python
df.pivot_table(index='Underaged',columns='Sex',values='Survived',aggfunc='mean')
```

```python
import datetime
```

```python
dt = datetime.datetime(year=2020,month=10,day=5,hour=11,minute=9)
dt
```

```python
print(dt)
```

```python
ts = pd.Timestamp('2020-10-05')
ts
```

```python
ts.month
```

```python
ts.day
```

```python
ts + pd.Timedelta('5 days')
```

```python
pd.to_datetime('2020-10-05')
```

```python
pd.to_datetime('10/05/2020')
```

```python
s = pd.Series(['2020-10-01 00:00:00','2020-10-03 00:00:00','2020-10-05 00:00:00'])
s
```

```python
ts = pd.to_datetime(s)
ts
```

```python
ts.dt.hour
```

```python
ts.dt.weekday
```

```python
pd.Series(pd.date_range(start='2020-10-05',periods=10,freq='12H'))
```

```python
data = pd.read_csv('./data/flowdata.csv')
data.head()
```

```python
data['Time'] = pd.to_datetime(data['Time'])
data = data.set_index('Time')
data
```

```python
data.index
```

```python
data = pd.read_csv('./data/flowdata.csv', index_col=0, parse_dates=True)
data.head()
```

```python
data[pd.Timestamp('2012-01-01 09:00'):pd.Timestamp('2012-01-01 19:00')]
```

```python
data[('2012-01-01 09:00'):('2012-01-01 19:00')]
```

```python
data.tail(10)
```

```python
data['2013']
```

```python
data['2012-01':'2012-03']
```

```python
data[data.index.month==1]
```

```python
data.between_time('08:00','12:00')
```

```python
data.head()
```

```python
data.resample('D').mean().head()
```

```python
data.resample('D').max().head()
```

```python
data.resample('3D').mean().head()
```

```python
data.resample('M').mean().head()
```

```python
import matplotlib.pyplot as plt
data.resample('M').mean().plot()
plt.show()
```

```python
data = pd.DataFrame({'group':['a','a','a','b','b','b','c','c','c'],
                    'data':[4,3,2,1,12,3,4,5,7]})
data
```

```python
data.sort_values(by=['group','data'],ascending=[False,True],inplace=True)
data
```

```python
data = pd.DataFrame({'k1':['one']*3+['two']*4,
                    'k2':[3,2,1,3,3,4,4]})
data
```

```python
data.sort_values(by='k2')
```

```python
data.drop_duplicates()
```

```python
data.drop_duplicates(subset='k1')
```

```python
data = pd.DataFrame({'food':['A1','A2','B1','B2','B3','C1','C2'],'data':[1,2,3,4,5,6,7]})
data
```

```python
def food_map(series):
    if series['food'] == 'A1':
        return 'A'
    elif series['food'] == 'A2':
        return 'A'
    elif series['food'] == 'B1':
        return 'B'
    elif series['food'] == 'B2':
        return 'B'
    elif series['food'] == 'B3':
        return 'B'
    elif series['food'] == 'C1':
        return 'C'
    elif series['food'] == 'C2':
        return 'C'

data['food_map'] = data.apply(food_map, axis='columns')
data
```

```python
food2Upper = {
    'A1':'A',
    'A2':'A',
    'B1':'B',
    'B2':'B',
    'B3':'B',
    'C1':'C',
    'C2':'C'
}
data['upper']=data['food'].map(food2Upper)
data
```

```python
titanic = pd.read_csv('./data/titanic.csv')
titanic.head()
```

```python
def hundredth_row(columns):
    item = columns.iloc[99]
    return item

hundredth_row = titanic.apply(hundredth_row)
hundredth_row
```

```python
def not_null_count(columns):
    return len(columns[pd.isnull(columns)])

columns_null_count = titanic.apply(not_null_count)
columns_null_count
```

```python
def which_class(row):
    pclass = row['Pclass']
    if pd.isnull(pclass):
        return 'Unknow'
    elif pclass == 1:
        return 'First class'
    elif pclass == 2:
        return 'Second class'
    elif pclass == 3:
        return 'Third class'

classes = titanic.apply(which_class, axis=1)
classes
```

```python
def is_minor(row):
    if row['Age'] < 18:
        return True
    else:
        return False
    
minors = titanic.apply(is_minor, axis=1)
minors
```

```python
df = pd.DataFrame({'data1':np.random.randn(5),'data2':np.random.randn(5)})
df2 = df.assign(ration=df['data1']/df['data2'])
df2
```

```python
df2.drop('ration',axis='columns',inplace=True)
df2
```

```python
df2['ration']=df['data1']/df['data2']
df2
```

```python
data = pd.Series([1,2,3,4,5,6,7,8,9])
data
```

```python
data.replace(9, np.nan, inplace=True)
data
```

```python
ages = [15,18,20,21,22,34,41,52,63,79]
bins = [10,40,80]
bins_res = pd.cut(ages,bins)
bins_res
```

```python
pd.value_counts(bins_res)
```

```python
pd.cut(ages,[10,30,50,80])
```

```python
group_names = ['Yonth','Mille','Old']
pd.value_counts(pd.cut(ages,[10,20,50,80],labels=group_names))
```

```python
df = pd.DataFrame([range(3),[0, np.nan, 0], [0,0,np.nan],range(3)])
df
```

```python
df.isnull()
```

```python
df.isnull().any()
```

```python
df.isnull().any(axis=1)
```

```python
df.fillna(5)
```

```python
df[df.isnull().any(axis=1)]
```

```python
s = pd.Series(['A','b','B','gaer','AGER',np.nan])
s
```

```python
s.str.lower()
```

```python
s.str.upper()
```

```python
s.str.len()
```

```python
index = pd.Index(['  left','  middle  ','right   '])
index
```

```python
index.str.strip()
```

```python
index.str.lstrip()
```

```python
index.str.rstrip()
```

```python
df = pd.DataFrame(np.random.randn(3,2),columns=['A a','B b'],index=range(3))
df
```

```python
df.columns = df.columns.str.replace(' ','_')
df
```

```python
df = pd.DataFrame(np.random.randn(3,2),columns=['A a','B b'],index=range(3))
df.columns = df.columns.str.replace(' ','')
df
```

```python
s = pd.Series(['a_b_C', 'c_d_e', 'f_g_h'])
s
```

```python
s.str.split('_')
```

```python
s.str.split('_', expand=True)
```

```python
s.str.split('_', expand=True, n=1)
```

```python
s = pd.Series(['A', 'Aas', 'Afgew', 'Ager', 'Agre', 'Ager'])
s
```

```python
s.str.contains('Ag')
```

```python
s = pd.Series(['a','a|b','a|c'])
s
```

```python
s.str.get_dummies(sep='|')
```

