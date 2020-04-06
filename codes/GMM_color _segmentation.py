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

def generate_data():
    filename = r'.\data\img\\'
    with open(r'.\roi.json') as f:
        data = json.load(f)
    key = list(data['_via_img_metadata'].keys())
    pixels = np.array([[0,0,0]])
    mean = 0
    for i in key:
        file = filename+data['_via_img_metadata'][i]['filename']
        regions = data['_via_img_metadata'][i]['regions']#[0]['shape_attributes']
        x = []
        y = []
        r = []
        for reg in regions:
            try:
                x.append(reg['shape_attributes']['cx'])
                y.append(reg['shape_attributes']['cy'])
                r.append(reg['shape_attributes']['r'])
            except:
                pass
        img = cv2.imread(file)
        blank = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        for j in range(len(x)):
            blank  = cv2.circle(blank, (x[j],y[j]),int(r[j]),1, -1)
        mask = cv2.bitwise_and(img, img, mask=blank)
        mask_ = np.any(mask!=[0,0,0], axis=-1)
        xcords, ycords = np.where(mask_==True)
        mean += len(xcords)
        pixels = np.vstack([pixels,mask[xcords, ycords]])
    print(mean/len(key))
    return pixels

def disp_hist(a, bins =50):
    b=a[:,0]
    g=a[:,1]
    r=a[:,2]
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.hist(r, bins)
    plt.title("Red Channel")
    plt.subplot(312)
    plt.hist(g, bins)
    plt.title("Green Channel")
    plt.subplot(313)
    plt.hist(b, bins)
    plt.title("Blue Chanel")
    plt.show()

pixels = generate_data()

gmm = GMM(3,200,3, if_priors=True)
gmm.fit(pixels[1:])


np.save('dataset.npy', pixels[1:])
np.save('means.npy', gmm.means)
np.save('covar.npy', gmm.sigma)

##########################################################
## Link to Results https://drive.google.com/file/d/1Z9uDHLfe75MrKG30JxXfgh1jMac8OpRA/view?usp=sharing
##########################################################