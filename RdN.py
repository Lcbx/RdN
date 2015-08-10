import cPickle, gzip, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, time

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set

#train_set_x = np.reshape(train_set_x,(50000,1,784))
#valid_set_x = np.reshape(valid_set_x,(10000,1,784))

# image = np.reshape(train_set_x[0], (28, 28))
# print np.shape(image)
# imgplot = plt.imshow(image)
# plt.imshow(image, cmap = cm.Greys_r)
# plt.show()


#the RdN
class layer(object):
    def __init__(self, nIn, nOut): # Notre methode constructeur        
        self.w = np.random.random_integers(-3,3, (nIn,nOut))
        self.b = np.random.random_integers(-3,3, (1,nOut))
    def fprop(self, x):
    	z = np.dot(x, self.w) + self.b
    	return z
    def update(self, lr, x, dEdy):
    	dEdw = np.dot(x.T, dEdy)
    	dEdb = np.mean(dEdy, axis = 0)    	
    	self.w = self.w - dEdw * lr
    	self.b = self.b - dEdb * lr
    	dEdx = np.dot(dEdy,self.w.T)
    	return dEdx
    
class MLP(object):
    def __init__(self):
        self.L = layer(784,100)
        self.M = layer(100,10)
    def fprop(self, x):
        self.cache = self.L.fprop(x)
        return self.M.fprop(self.cache)
    def update(self, lr, x, t):
        dEdy = 2*(self.fprop(x) - np.eye(10)[t])
        self.L.update(lr, x, self.M.update(lr, self.cache, dEdy))
        return
    def error(self, x, t):
        z = np.argmax(self.fprop(x), axis = 1)
        errors = np.sum(z!=t)
        return errors

     
RdN = MLP()
batch = 1000 # nbr dexemples appris a la fois
# mesuring time
start = time.time()
for i in range(6):
    # training
    for j in range(50000/batch):
    	RdN.update(.00001, train_set_x[j*batch:(j+1)*batch], train_set_y[j*batch:(j+1)*batch])
    # mesuring error rate
    error_count = 0
    for j in range(10000/batch):
        error_count += RdN.error(valid_set_x[j*batch:(j+1)*batch], valid_set_y[j*batch:(j+1)*batch])
    error_rate = error_count / 10000.
    # report
    print "validation error rate : " + str(error_rate) + " epoch " + str(i) 
print " execution time : " + str(time.time() - start)
    
    

    	
