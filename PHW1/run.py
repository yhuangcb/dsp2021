from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import os

mnist = fetch_openml('mnist_784' , as_frame= False)
output_dir = "./fig/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
##Q2

fives_data = mnist['data'][mnist['target']=='5']
centered_5 = fives_data - fives_data.mean(axis=0)
scatter_5 = centered_5.T @ centered_5
w, v = np.linalg.eig(scatter_5)

for i in range(3):
    plt.subplot(131+i)
    plt.imshow(np.real(v.T[i].reshape(28,28)), 'gray')
    plt.title('λ=' + str(format(np.real(w[i]), '10.2E')))
    plt.axis('off')
plt.savefig(output_dir + "Q2.jpg")

##Q3

plt.subplot(151)
plt.imshow(fives_data[0].reshape(28,28), 'gray')
plt.title('Original 5')
plt.axis('off')

for i, d in enumerate([3,10,30,100]):
    plt.subplot(152+i)
    temp = (v.T[0:d] @ centered_5[0]).reshape((-1,1)) * v.T[0:d]
    plt.imshow(np.real((temp.sum(axis=0)+fives_data.mean(axis=0)).reshape(28,28)), 'gray')
    plt.title('5 with ' + str(d)+'d')
    plt.axis('off')
plt.savefig(output_dir + "Q3.jpg")


##Q4

first_10k = mnist['data'][0:10000]
first_10k_label = mnist['target'][0:10000]
first_10k_1 = first_10k[first_10k_label == '1']
first_10k_3 = first_10k[first_10k_label == '3']
first_10k_6 = first_10k[first_10k_label == '6']
first_10k_136 = np.concatenate((first_10k_1,first_10k_3,first_10k_6),axis=0)

def centered_PCA(data):
    # input data should be a ndarray
    centered = data - data.mean(axis=0)
    scatter = centered.T @ centered
    w, v = np.linalg.eig(scatter)
    return w, v

def reconstruct(data, v, d):
    return np.real(v.T[0:d] @ data.T)

W, V = centered_PCA(first_10k_136)
points = reconstruct(first_10k_136-first_10k_136.mean(axis=0), V, 2) # [0]=x [1]=y
colors = np.array(['red']*len(first_10k_1) + ['green']*len(first_10k_3) + ['blue']*len(first_10k_6))
plt.clf()
plt.scatter(points[0], points[1], c=colors)
#plt.show()
plt.savefig(output_dir + "Q4.jpg")

##Q5

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

prenorm10k = first_10k / np.linalg.norm(first_10k, axis=1).reshape((-1,1))
OMP_target = mnist['data'][10000].reshape(-1,1)
temp = OMP(prenorm10k, OMP_target, 5)
for i in range(5):
    plt.subplot(151+i)
    plt.imshow(prenorm10k[temp[i]].reshape(28,28), 'gray')
    plt.title('base' + str(i+1))
    plt.axis('off')
plt.savefig(output_dir + "Q5.jpg")

##Q6

OMP_target_2 = mnist['data'][10001].reshape(-1,1)
sp_5 = OMP(prenorm10k, OMP_target_2, 5)
sp_10 = OMP(prenorm10k, OMP_target_2, 10)
sp_40 = OMP(prenorm10k, OMP_target_2, 40)
sp_200 = OMP(prenorm10k, OMP_target_2, 200)

re_5 = OMP_reconstruct(prenorm10k[sp_5].T, OMP_target_2)
re_10 = OMP_reconstruct(prenorm10k[sp_10].T, OMP_target_2)
re_40 = OMP_reconstruct(prenorm10k[sp_40].T, OMP_target_2)
re_200 = OMP_reconstruct(prenorm10k[sp_200].T, OMP_target_2)
re = [re_5, re_10, re_40, re_200]

plt.subplot(151)
plt.imshow(OMP_target_2.reshape(28,28), 'gray')
plt.title('L-2=0')
plt.axis('off')
for i in range(4):
    plt.subplot(152+i)
    plt.imshow(re[i].reshape(28,28), 'gray')
    L2err = np.linalg.norm(re[i]-OMP_target_2)
    L2err = float("{:.1f}".format(L2err))
    plt.title('L-2='+str(L2err))
    plt.axis('off')
plt.savefig(output_dir + "Q6.jpg")





