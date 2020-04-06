import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import json

class GMM():
    def __init__(self, num_clusters, num_iters, dims, if_priors = False):
        self.num_iters = num_iters
        self.dims = dims
        self.num_clusters = num_clusters
        self.means = np.random.randint(0,255, size=(num_clusters,dims))
        self.sigma = []
        for i in range(num_clusters):
            vec = np.random.random(size=(dims))*100+100
            self.sigma.append(np.diag(vec))
        self.if_priors = if_priors
        self.priori = np.array([1/self.num_clusters]*self.num_clusters)
#         print(self.sigma)
        
    def EM(self,data):
        
#         print(self.means,'\n' ,self.sigma, self.priori)
        data = np.array(data)
        
        # E step
        likilihood = np.empty(shape = (self.num_clusters, len(data)))
        for i in range(self.num_clusters):
            norm_dist = multivariate_normal(self.means[i], self.sigma[i])
#             likilihood[i,:] = norm_dist.pdf(data*self.priori[i])
            for j in range(len(data)):
                likilihood[i,j] = norm_dist.pdf(data[j])*self.priori[i]
        weighted_sum = np.sum(likilihood, axis = 0)
        for i in range(self.num_clusters):
            for j in range(len(data)):
                likilihood[i,j] = likilihood[i,j]/weighted_sum[j]
        total_probs = np.sum(likilihood, axis = 1)
        
        # M step
        for i in range(self.num_clusters):
            add = 0
            for j in range(len(data)):
                add+= likilihood[i,j]*data[j]
            self.means[i] = add/total_probs[i]
            
#         print(total_probs)
        for i in range(self.num_clusters):
            add = np.zeros(shape = (self.dims, self.dims))
            for j in range(len(data)):
                diff = (data[j] - self.means[i]).reshape(1, self.dims)
                if(self.dims==1):
                    add+= likilihood[i,j]*diff.T.dot(diff)
                else:
                    ans = np.dot(np.multiply(diff.T, likilihood[i,j].T), diff)
                    add+= np.diag(np.diag(ans)) #likilihood[i,j]*diff.T.dot(diff)
            self.sigma[i] = (add/total_probs[i])**0.5
                
        if self.if_priors:
            for i in range(self.num_clusters):
                self.priori[i] = np.sum(total_probs[i])/len(data)
        
    def fit(self, data, iters = None):
        if iters!= None:
            self.num_iters = iters
        for i in range(self.num_iters):
            self.EM(data)


a = np.random.normal(0, 2, size=(100,1))
b = np.random.normal(3, 0.5, size=(100,1))
c = np.random.normal(6, 3, size=(100,1))
data = np.concatenate((a,b,c))

x = np.linspace(-5,12,1001)
y = multivariate_normal(0,2).pdf(x)+multivariate_normal(3,0.5).pdf(x)+multivariate_normal(6,3).pdf(x)

gmm = GMM(3,200,1)
gmm.fit(data)

y_  = multivariate_normal(gmm.means[0],gmm.sigma[0]).pdf(x)\
    +multivariate_normal(gmm.means[1],gmm.sigma[1]).pdf(x) \
    + multivariate_normal(gmm.means[2],gmm.sigma[2]).pdf(x)
plt.scatter(data, [0.1]*len(data),color = 'r', s=0.1, label = "Data Points")
plt.plot(x,y,'b',label = "original")
plt.plot(x,y_,'g', label = "predicted")
plt.legend()
plt.show()