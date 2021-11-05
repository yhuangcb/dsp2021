#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import os


# In[2]:


mnist = fetch_openml('mnist_784' , as_frame= False)


# In[6]:


fig = plt.figure(figsize=(15,3))
fig.patch.set_facecolor('white')
for i in range(9):
    plt.subplot(191+i)
    plt.imshow(mnist['data'][i].reshape(28,28), 'gray')
    plt.title(mnist['target'][i])
    plt.axis('off')


# ## PCA
# ### Q1

# In[7]:


plt.imshow((mnist['data'].mean(axis=0)).reshape(28,28), 'gray')


# In[8]:


fives_data = mnist['data'][mnist['target']=='5']
centered_5 = fives_data - fives_data.mean(axis=0)
scatter_5 = centered_5.T @ centered_5


# In[12]:


w, v = np.linalg.eig(scatter_5)


# In[15]:


for i in range(3):
    plt.subplot(131+i)
    plt.imshow(np.real(v.T[i].reshape(28,28)), 'gray')
    plt.title('λ=' + str(format(np.real(w[i]), '10.2E')))
    plt.axis('off')


# In[16]:


d = 100

temp = (v.T[0:d] @ centered_5[0]).reshape((-1,1)) * v.T[0:d]


# In[19]:


plt.imshow(np.real((temp.sum(axis=0)+fives_data.mean(axis=0)).reshape(28,28)), 'gray')


# In[20]:


plt.imshow(np.real((centered_5[0]+fives_data.mean(axis=0)).reshape(28,28)), 'gray')


# In[21]:


first_10k = mnist['data'][0:10000]
first_10k_label = mnist['target'][0:10000]


# In[99]:


first_10k_1 = first_10k[first_10k_label == '1']
first_10k_3 = first_10k[first_10k_label == '3']
first_10k_6 = first_10k[first_10k_label == '6']
first_10k_136 = np.concatenate((first_10k_1,first_10k_3,first_10k_6),axis=0)


# In[3]:


def centered_PCA(data):
    # input data should be a ndarray
    centered = data - data.mean(axis=0)
    scatter = centered.T @ centered
    w, v = np.linalg.eig(scatter)
    return w, v

def reconstruct(data, v, d):
    return np.real(v.T[0:d] @ data.T)


# In[100]:


W, V = centered_PCA(first_10k_136)
#print(reconstruct(first_10k_136[0:3], V, 2))


# In[101]:


points = reconstruct(first_10k_136-first_10k_136.mean(axis=0), V, 2) # [0]=x [1]=y
colors = np.array(['red']*len(first_10k_1) + ['green']*len(first_10k_3) + ['blue']*len(first_10k_6))


# In[102]:


plt.scatter(points[0], points[1], c=colors)
plt.show()


# In[192]:


OMP_target = mnist['data'][10000].reshape(-1,1)


# In[168]:


b1 = prenorm10k[3606].reshape((-1,1))
MTMinv = np.linalg.inv(b1.T @ b1)
u = MTMinv @ b1.T @ OMP_target
print(u)
#print(np.linalg.inv(b1.T @ b1))

#(b1 @ b1.T).shape
#np.dot(b1, OMP_target) * b1


# In[134]:


prenorm10k = first_10k / np.linalg.norm(first_10k, axis=1).reshape((-1,1))


# In[40]:


def largest(product, Lambda):
    #print(product)
    a = np.argsort(product.reshape((-1)))
    #print(a)
    if not Lambda:
        return a[-1]
    while a[-1] in Lambda:
        a = a[:-1]
    return a[-1]

def OMP_step(B, last_r, Lambda):
    #sl = np.argmax(B @ OMP_target)
    sl = largest(B @ last_r, Lambda)
    L = Lambda + [sl]
    #print(L)
    
    b = B[L].reshape((-1,len(L)))
    c = np.linalg.pinv(b.T @ b) @ b.T @ last_r
    r = last_r - b@c
    
    return r, sl



def OMP(B, OMP_target, sparsity):
    Lambda = []
    last_r = OMP_target
    for _ in range(sparsity):
        last_r, sl = OMP_step(B, last_r, Lambda)
        Lambda.append(sl)
        
    return Lambda
        
    
def OMP_reconstruct(b, last_r):
    
    c = np.linalg.pinv(b.T @ b) @ b.T @ last_r
    return b@c


# In[ ]:


temp = OMP(prenorm10k, OMP_target, 5)
print(temp)
#plt.imshow(temp.reshape(28,28), 'gray')


# In[221]:


for i in range(5):
    plt.subplot(151+i)
    plt.imshow(prenorm10k[temp[i]].reshape(28,28), 'gray')
    plt.title('base' + str(i))
    plt.axis('off')


# In[222]:


OMP_target_2 = mnist['data'][10001].reshape(-1,1)
sp_5 = OMP(prenorm10k, OMP_target_2, 5)
sp_10 = OMP(prenorm10k, OMP_target_2, 10)
sp_40 = OMP(prenorm10k, OMP_target_2, 40)
sp_200 = OMP(prenorm10k, OMP_target_2, 200)


# In[243]:



re_5 = OMP_reconstruct(prenorm10k[sp_5].T, OMP_target_2)
re_10 = OMP_reconstruct(prenorm10k[sp_10].T, OMP_target_2)
re_40 = OMP_reconstruct(prenorm10k[sp_40].T, OMP_target_2)
re_200 = OMP_reconstruct(prenorm10k[sp_200].T, OMP_target_2)
re = [re_5, re_10, re_40, re_200]

for i in range(4):
    plt.subplot(141+i)
    plt.imshow(re[i].reshape(28,28), 'gray')
    #plt.title(str(np.linalg.norm(re[i]-OMP_target_2)))
    plt.axis('off')
    print(np.linalg.norm(re[i]-OMP_target_2))


# In[240]:


plt.imshow(OMP_target_2.reshape(28,28), 'gray')


# In[3]:


All_8 = mnist['data'][mnist['target']=='8']
centered_8 = All_8 - All_8.mean(axis=0)


# In[38]:


W, V = centered_PCA(All_8)
#print(reconstruct(All_8[-1], V, 5))
re_last_8 = reconstruct(All_8[-1], V, 5).reshape((-1,1))
#print(V.T[0:5].shape, re_last_8.shape)
re_last_8_image = np.real(re_last_8.T @ V.T[0:5])
#print(re_last_8_image)
plt.imshow((re_last_8_image+All_8.mean(axis=0)).reshape(28,28), 'gray')


# In[43]:


temp3 = OMP(All_8[:-1], All_8[-1], 5)
re_last8_OMP = OMP_reconstruct(All_8[temp3].T, All_8[-1])
plt.imshow(re_last8_OMP.reshape(28,28), 'gray')


# In[17]:


from sklearn.linear_model import Lasso

clf = Lasso(alpha=1)
clf.fit(All_8[:-1].T, All_8[-1])
#print(clf.coef_)


# In[18]:


yee = clf.coef_ @ All_8[:-1]
plt.imshow(yee.reshape(28,28), 'gray')
print(np.count_nonzero(clf.coef_))


# In[19]:


plt.imshow((yee-All_8[-1]).reshape(28,28), 'gray')


# In[111]:


A = All_8[:-1].T
y = All_8[-1]
#x = np.random.rand(6824)
x = np.zeros(6824)


# In[133]:


def lasso_update(A, i, alpha, x, y):
    x_i = np.zeros_like(x)
    x_i[i] = x[i]
    x_no_i = np.copy(x)
    x_no_i[i] = 0

    A_i = np.zeros_like(A)
    A_i = A_i.T
    A_i[i] = A.T[i]
    A_i = A_i.T
    A_no_i = np.copy(A)
    A_no_i = A_no_i.T
    A_no_i[i] = 0
    A_no_i = A_no_i.T
    #value = np.linalg.norm(A_i.T @(y - A_no_i @ x_no_i)) / np.linalg.norm(A_i.T @ A_i)
    value = ((A_i.T @(y - A_no_i @ x_no_i)) / np.linalg.norm(A_i.T @ A_i)).sum()
    threshold = alpha / np.power(np.linalg.norm(A_i), 2)
    if np.absolute(value) > threshold:
        #print(value)
        return value
    else:
        return 0

def my_lasso(A, y, alpha, iteration):
    # A should have shape (n_features, N)
    # for example (784, 6824)
    N = A.shape[1]
    x = np.zeros(N)
    count = 0
    for _ in range(iteration):
        for i in range(N):
            #print(A, i, alpha, x, y)
            x[i] = lasso_update(A, i, alpha, x, y)
            print(i)
            
    return x
        


# In[134]:


A = All_8[:-1].T
y = All_8[-1]
las = my_lasso(A, y, 0, 1)
#print(las)


# In[135]:


plt.imshow((A @ las).reshape(28,28), 'gray')

