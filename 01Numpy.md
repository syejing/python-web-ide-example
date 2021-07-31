```python
import numpy as np
```

```python
array = np.array([1,2,3,4,5])
print(type(array))
```

```python
array2 = array + 1
array2
```

```python
array2 + array
```

```python
array2 * array
```

```python
array[0]
```

```python
array[3]
```

```python
array
```

```python
array.shape
```

```python
np.array([[1,2,3],[4,5,6]])
```

```python
np.array([[[1,2,3],[4,5,6]]])
```

```python
np.array([[[1,2,3],[4,5,6]]]).shape
```

```python
np.array([1,2,3,5,6])
```

```python
np.array([1,2,3,5,6.0])
```

```python
test_array = np.array([1,2,3,5,'6'])
test_array
```

```python
type(test_array)
```

```python
test_array.dtype
```

```python
test_array.shape
```

```python
test_array.itemsize
```

```python
np.size(test_array)
```

```python
np.shape(test_array)
```

```python
test_array.ndim
```

```python
test_array
```

```python
test_array = np.array([1,2,3,4,5])
test_array
```

```python
test_array.fill(0)
test_array
```

```python
test_list = [1,2,3,5,6]
test_array = np.array(test_list)
test_array
```

```python
test_array[0]
```

```python
test_array[1:3]
```

```python
test_array[-2:]
```

```python
test_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
test_array
```

```python
test_array.shape
```

```python
test_array.size
```

```python
test_array.ndim
```

```python
test_array[1,1]
```

```python
test_array[1,1] = 10
test_array
```

```python
test_array[1]
```

```python
test_array[:, 1]
```

```python
test_array[0,0:2]
```

```python
test_array
```

```python
test_array2 = test_array
test_array2
```

```python
test_array2[1,1] = 100
test_array2
```

```python
test_array
```

```python
test_array2 = test_array.copy()
test_array2
```

```python
test_array2[1,1] = 10
test_array2
```

```python
test_array
```

```python
test_array = np.arange(0,100,10)
test_array
```

```python
mask = np.array([0,0,0,1,1,1,0,0,1,1],dtype=bool)
mask
```

```python
test_array[mask]
```

```python
random_array = np.random.rand(10)
random_array
```

```python
mask = random_array > 0.5
mask
```

```python
random_array[mask]
```

```python
import os
print(os.path.abspath('.'))
```

```python
test_array = np.array([10,20,30,40,50])
test_array > 30
```

```python
np.where(test_array > 30)
```

```python
test_array[np.where(test_array > 30)]
```

```python
test_array = np.array([1,2,3,4,5],dtype=np.float32)
test_array
```

```python
test_array.dtype
```

```python
test_array.nbytes
```

```python
test_array = np.array([1,10,3.5,'str'],dtype=np.object)
test_array
```

```python
test_array = np.array([1,2,3,4,5])
test_array
```

```python
test_array.astype(np.float32)
```

```python
test_array = np.array([[1,2,3],[4,5,6]])
test_array
```

```python
np.sum(test_array)
```

```python
np.sum(test_array, axis=0)
```

```python
np.sum(test_array, axis=1)
```

```python
test_array.ndim
```

```python
test_array.prod(axis=0)
```

```python
test_array.prod(axis=1)
```

```python
test_array.min()
```

```python
test_array.min(axis=0)
```

```python
test_array.min(axis=1)
```

```python
test_array.max()
```

```python
test_array.argmin()
```

```python
test_array.argmin(axis=0)
```

```python
test_array.argmin(axis=1)
```

```python
test_array.argmax()
```

```python
test_array.mean()
```

```python
test_array.std()
```

```python
test_array.std(axis=1)
```

```python
test_array.var()
```

```python
test_array
```

```python
test_array.clip(2,4)
```

```python
test_array = np.array([1.2,3.56,6.41])
test_array
```

```python
test_array.round()
```

```python
test_array.round(decimals=1)
```

```python
test_array = np.array([[1.5,1.3,7.5],[5.6,7.8,1.2]])
test_array
```

```python
np.sort(test_array)
```

```python
np.sort(test_array, axis=0)
```

```python
test_array
```

```python
np.argsort(test_array)
```

```python
test_array = np.linspace(0,10,10)
test_array
```

```python
values = np.array([2.5,6.5,9.5])
values
```

```python
np.searchsorted(test_array,values)
```

```python
test_array = np.array([[1,0,6],[1,7,0],[2,3,1],[2,4,0]])
test_array
```

```python
index = np.lexsort([-1*test_array[:,0],test_array[:,2]])
index
```

```python
test_array[index]
```

```python
test_array = np.arange(10)
test_array
```

```python
test_array.shape
```

```python
test_array.shape = 2,5
test_array
```

```python
test_array.reshape(1,10)
```

```python
test_array = np.arange(10)
test_array.shape
```

```python
test_array = test_array[np.newaxis,:]
print(test_array)
test_array.shape
```

```python
test_array = np.arange(10)
test_array.shape
```

```python
test_array = test_array[:,np.newaxis]
print(test_array)
test_array.shape
```

```python
test_array = test_array[:,np.newaxis,np.newaxis]
test_array.shape
```

```python
test_array
```

```python
test_array = test_array.squeeze()
test_array.shape
```

```python
test_array.shape = 2,5
test_array
```

```python
test_array.transpose()
```

```python
test_array.T
```

```python
a = np.array([[123,456,678],[3214,456,134]])
a
```

```python
b = np.array([[1235,3124,432],[43,13,134]])
b
```

```python
c = np.concatenate((a,b))
c
```

```python
np.concatenate((a,b), axis=0)
```

```python
c = np.concatenate((a,b), axis=1)
c
```

```python
c.shape
```

```python
np.vstack((a,b))
```

```python
np.hstack((a,b))
```

```python
a
```

```python
a.flatten()
```

```python
np.array([1,2,3])
```

```python
np.arange(10)
```

```python
np.arange(2,20,2)
```

```python
np.arange(2,20,2,dtype=np.float32)
```

```python
np.linspace(0,10,10)
```

```python
np.logspace(0,1,5)
```

```python
x = np.linspace(-10,10,5)
x
```

```python
np.zeros(3)
```

```python
np.zeros((3,3))
```

```python
np.ones((3,3))
```

```python
np.ones((3,3)) * 8
```

```python
np.ones((3,3), dtype = np.float32)
```

```python
a = np.empty(6)
a.shape
```

```python
a
```

```python
a.fill(1)
a
```

```python
test_array = np.array([1,2,3,4])
test_array
```

```python
np.zeros_like(test_array)
```

```python
np.ones_like(test_array)
```

```python
np.identity(5)
```

```python
x = np.array([5,5])
y = np.array([2,2])
```

```python
np.multiply(x,y)
```

```python
np.dot(x, y)
```

```python
y.shape = 1,2
y
```

```python
x.shape = 2,1
x
```

```python
print(x)
print(y)
```

```python
np.dot(x,y)
```

```python
np.dot(y,x)
```

```python
x = np.array([1,1,1,2])
y = np.array([1,1,1,4])
x == y
```

```python
np.logical_and(x,y)
```

```python
np.logical_or(x,y)
```

```python
np.logical_not(x,y)
```

```python
np.random.rand(3,2)
```

```python
np.random.randint(10, size=(5,4))
```

```python
np.random.rand()
```

```python
np.random.random_sample()
```

```python
np.random.randint(0,10,3)
```

```python
mu, sigma = 0, 0.1
np.random.normal(mu, sigma, 10)
```

```python
np.set_printoptions(precision=2)
```

```python
mu, sigma = 0, 0.1
np.random.normal(mu, sigma, 10)
```

```python
test_array = np.arange(10)
test_array
```

```python
np.random.shuffle(test_array)
```

```python
test_array
```

```python
np.random.seed(100)
mu, sigma = 0, 0.1
np.random.normal(mu, sigma, 10)
```

```python
data = []
with open('./data/test1.txt') as f:
    for line in f.readlines():
        fileds = line.split()
        cur_data = [float(x) for x in fileds]
        data.append(cur_data)
data = np.array(data)
data
```

```python
data = np.loadtxt('./data/test1.txt')
data
```

```python
data = np.loadtxt('./data/test2.txt', delimiter=',')
data
```

```python
data = np.loadtxt('./data/test3.txt',delimiter=',',skiprows=1)
data
```

```python
test_array = np.array([[1,2,3],[4,5,6]])
test_array
```

```python
np.savetxt('./data/test4.txt', test_array)
```

```python
np.savetxt('./data/test4.txt', test_array, fmt='%d')
```

```python
np.savetxt('./data/test4.txt', test_array, fmt='%d', delimiter=',')
```

```python
np.savetxt('./data/test4.txt', test_array, fmt='%.2f', delimiter=',')
```

```python
test_array = np.array([[1,2,3],[4,5,6]])
np.save('./data/test_array.npy', test_array)
```

```python
test_array2 = np.load('./data/test_array.npy')
test_array2
```

```python
test_array2 = np.arange(10)
test_array2
```

```python
np.savez('./data/test.npz', a=test_array, b=test_array2)
```

```python
data = np.load('./data/test.npz')
```

```python
data.keys()
```

```python
data['a']
```

