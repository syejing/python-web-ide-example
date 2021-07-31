```python
import numpy as np
import pandas as pd
from scipy import stats,integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
def siplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x, np.sin(x+i*.5) *(7-i)*flip)
    plt.show()
siplot()
```

```python
sns.set()
siplot()
```

```python
sns.set_style('whitegrid')
data = np.random.normal(size=(20,6)) + np.arange(6) / 2
sns.boxplot(data=data)
plt.show()
```

```python
sns.set_style('dark')
siplot()
```

```python
sns.set_style('white')
siplot()
```

```python
sns.set_style('ticks')
siplot()
```

```python
sns.violinplot(data)
sns.despine(offset=10)
plt.show()
```

```python
sns.set_style('whitegrid')
sns.boxplot(data=data, palette='deep')
plt.show()
```

```python
sns.set_style('whitegrid')
sns.boxplot(data=data, palette='deep')
sns.despine(left=True)
plt.show()
```

```python
with sns.axes_style('darkgrid'):
    plt.subplot(211)
    siplot()
plt.subplot(212)
siplot(-1)
```

```python
sns.set()
```

```python
sns.set_context('paper')
plt.figure(figsize=(8,6))
siplot()
```

```python
sns.set_context('talk')
plt.figure(figsize=(8,6))
siplot()
```

```python
sns.set_context('poster')
plt.figure(figsize=(8,6))
siplot()
```

```python
sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth":2.5})
plt.figure(figsize=(8,6))
siplot()
```

```python
sns.set(rc={"figure.figsize":(6,6)})
```

```python
current_palette = sns.color_palette()
sns.palplot(current_palette)
plt.show()
```

```python
sns.palplot(sns.color_palette('hls', 8))
plt.show()
```

```python
sns.palplot(sns.color_palette('hls', 12))
plt.show()
```

```python
data = np.random.normal(size=(20,8)) + np.arange(8)/2
sns.boxplot(data=data,palette=sns.color_palette('hls',8))
plt.show()
```

```python
sns.palplot(sns.hls_palette(8, l=.3, s=.8))
plt.show()
```

```python
sns.palplot(sns.color_palette('Paired', 10))
plt.show()
```

```python
plt.plot([0,1],[0,1],sns.xkcd_rgb['pale red'],lw=3)
plt.plot([0,1],[0,2],sns.xkcd_rgb['medium green'],lw=3)
plt.plot([0,1],[0,3],sns.xkcd_rgb['denim blue'],lw=3)
plt.show()
```

```python
colors = ['windows blue','amber','greyish','faded green','dusty purple']
sns.palplot(sns.xkcd_palette(colors))
plt.show()
```

```python
sns.palplot(sns.color_palette('Blues'))
plt.show()
```

```python
sns.palplot(sns.color_palette('BuGn_r'))
plt.show()
```

```python
sns.palplot(sns.color_palette('cubehelix',8))
plt.show()
```

```python
sns.palplot(sns.cubehelix_palette(8,start=.5,rot=-.75))
plt.show()
```

```python
sns.palplot(sns.cubehelix_palette(8,start=.75,rot=-.150))
plt.show()
```

```python
sns.palplot(sns.light_palette('green'))
plt.show()
```

```python
sns.palplot(sns.dark_palette('purple'))
plt.show()
```

```python
sns.palplot(sns.light_palette('navy', reverse=True))
plt.show()
```

```python
x,y = np.random.multivariate_normal([0,0],[[1,-.5],[-0.5,1]],size=300).T
pal = sns.dark_palette('green', as_cmap=True)
sns.kdeplot(x,y,cmap=pal)
plt.show()
```

```python
sns.palplot(sns.light_palette((210,90,60),input="husl"))
plt.show()
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))
```

```python
x = np.random.normal(size=100)
sns.distplot(x,kde=False)
plt.show()
```

```python
sns.distplot(x,bins=20,kde=False)
plt.show()
```

```python
x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)
plt.show()
```

```python
mean,cov = [0,1],[(1,.5),(.5,1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=['x','y'])
df
```

```python
sns.jointplot(x='x',y='y',data=df)
plt.show()
```

```python
x,y = np.random.multivariate_normal(mean,cov,1000).T
with sns.axes_style('white'):
    sns.jointplot(x=x,y=y,kind='hex',color='k')
plt.show()
```

```python
iris = sns.load_dataset('iris')
sns.pairplot(iris)
plt.show()
```

```python
sns.set(color_codes=True)
np.random.seed(sum(map(ord,'regression')))
```

```python
tips = sns.load_dataset('tips')
tips.head()
```

```python
sns.regplot(x='total_bill',y='tip',data=tips)
plt.show()
```

```python
sns.lmplot(x='total_bill',y='tip',data=tips)
plt.show()
```

```python
sns.regplot(data=tips,x='size',y='tip')
plt.show()
```

```python
sns.regplot(data=tips,x='size',y='tip',x_jitter=.05)
plt.show()
```

```python
sns.set(style='whitegrid', color_codes=True)
np.random.seed(sum(map(ord,"categorical")))

titanic = sns.load_dataset('titanic')
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
```

```python
sns.stripplot(x='day',y='total_bill',data=tips)
plt.show()
```

```python
sns.stripplot(x='day',y='total_bill',data=tips, jitter=False)
plt.show()
```

```python
sns.swarmplot(x='day',y='total_bill',data=tips)
plt.show()
```

```python
sns.swarmplot(x='day',y='total_bill',data=tips,hue='sex')
plt.show()
```

```python
sns.swarmplot(x='total_bill',y='day',data=tips,hue='time')
plt.show()
```

```python
sns.boxplot(x='day',y='total_bill',data=tips,hue='time')
plt.show()
```

```python
sns.violinplot(x='total_bill',y='day',data=tips,hue='time')
plt.show()
```

```python
sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)
plt.show()
```

```python
sns.violinplot(x='day',y='total_bill',data=tips,inner=None)
sns.swarmplot(x='day',y='total_bill',data=tips,color='w',alpha=.5)
plt.show()
```

```python
sns.barplot(x='sex',y='survived',data=titanic,hue='class')
plt.show()
```

```python
sns.pointplot(x='sex',y='survived',data=titanic,hue='class')
plt.show()
```

```python
sns.pointplot(x='class',y='survived',data=titanic,hue='sex',palette={"male":"g","female":"m"},marker=["^","o"],
              linestyles=["-","--"])
plt.show()
```

```python
sns.boxplot(data=iris,orient='h')
plt.show()
```

```python
sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker')
plt.show()
```

```python
sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker',kind='bar')
plt.show()
```

```python
sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker',col='time',kind='swarm')
plt.show()
```

```python
sns.factorplot(x='time',y='total_bill',data=tips,hue='smoker',col='day',kind='box',height=4.,aspect=.5)
plt.show()
```

```python
sns.set(style='whitegrid', color_codes=True)
np.random.seed(sum(map(ord,"axis_grids")))

tips = sns.load_dataset('tips')
tips.head()
```

```python
g = sns.FacetGrid(tips, col='time')
plt.show()
```

```python
g = sns.FacetGrid(tips, col='time')
g.map(plt.hist, 'tip')
plt.show()
```

```python
g = sns.FacetGrid(tips, col='sex', hue='smoker')
g.map(plt.scatter, 'total_bill', 'tip', alpha=.7)
g.add_legend()
plt.show()
```

```python
g = sns.FacetGrid(tips, row='smoker', col='time', margin_titles=True)
g.map(sns.regplot, 'size', 'total_bill', color='.3', fit_reg=False, x_jitter=.1)
plt.show()
```

```python
g = sns.FacetGrid(tips, col='day', height=4, aspect=.5)
g.map(sns.barplot, 'sex', 'total_bill')
plt.show()
```

```python
from pandas import Categorical
```

```python
ordered_days = tips.day.value_counts().index
ordered_days
```

```python
g = sns.FacetGrid(tips, row='day', row_order=ordered_days, height=1.7, aspect=4)
g.map(sns.boxplot, 'total_bill')
plt.show()
```

```python
ordered_days = Categorical(['Thur','Fri','Sat','Sun'])
g = sns.FacetGrid(tips, row='day', row_order=ordered_days, height=1.7, aspect=4)
g.map(sns.boxplot, 'total_bill')
plt.show()
```

```python
pal = dict(Lunch='seagreen', Dinner='gray')
g = sns.FacetGrid(tips,hue='time',palette=pal,height=5)
g.map(plt.scatter, 'total_bill', 'tip', s=50, alpha=.7, linewidth=.5, edgecolor='white')
g.add_legend()
plt.show()
```

```python
g = sns.FacetGrid(tips,hue='sex',palette='Set1',height=5, hue_kws={"marker":["^","v"]})
g.map(plt.scatter, 'total_bill', 'tip', s=100, linewidth=.5, edgecolor='white')
g.add_legend()
plt.show()
```

```python
with sns.axes_style('white'):
    g = sns.FacetGrid(tips, row='sex', col='smoker', margin_titles=True, height=2.5)
g.map(plt.scatter, 'total_bill', 'tip', color='#334488', edgecolor='white', lw=.5)
g.set_axis_labels('Total bill(US Dollars)','Tip')
g.set(xticks=[10,30,50],yticks=[2,6,10])
g.fig.subplots_adjust(wspace=.02,hspace=.02)
plt.show()
```

```python
iris = sns.load_dataset('iris')
g = sns.PairGrid(iris)
g.map(plt.scatter)
plt.show()
```

```python
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.show()
```

```python
g = sns.PairGrid(iris,hue='species')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
plt.show()
```

```python
g = sns.PairGrid(iris, vars=['sepal_length','sepal_width'],hue='species')
g.map(plt.scatter)
plt.show()
```

```python
g = sns.PairGrid(tips, hue='size', palette='GnBu_d')
g.map(plt.scatter, s=50, edgecolor='white')
g.add_legend()
plt.show()
```

```python
sns.set()
np.random.seed(0)
```

```python
uniform_data = np.random.rand(3,3)
uniform_data
```

```python
heatmap = sns.heatmap(uniform_data)
plt.show()
```

```python
sns.heatmap(uniform_data, vmin=0.2,vmax=0.5)
plt.show()
```

```python
normal_data = np.random.randn(3,3)
normal_data
```

```python
sns.heatmap(normal_data, center=0)
plt.show()
```

```python
flights = sns.load_dataset('flights')
flights.head()
```

```python
flights = flights.pivot('month','year','passengers')
flights
```

```python
sns.heatmap(flights)
plt.show()
```

```python
sns.heatmap(flights,annot=True,fmt='d')
plt.show()
```

```python
sns.heatmap(flights, linewidths=.5)
plt.show()
```

```python
sns.heatmap(flights, cmap='YlGnBu')
plt.show()
```

```python
sns.heatmap(flights, cbar=False)
plt.show()
```

