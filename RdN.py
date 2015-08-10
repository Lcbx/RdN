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

batch = 10
#the RdN
class layer(object):
    def __init__(self): # Notre methode constructeur        
        self.w = np.zeros((784,10))
        self.b = np.zeros((1,10))
    def fprop(self, x):
    	z = np.dot(x, self.w) + self.b
    	return z
    def update(self, lr, x, t):
    	dEdz = 2*(self.fprop(x) - np.eye(10)[t])
    	dEdw = np.dot(x.T, dEdz)
    	dEdb = np.mean(dEdz, axis = 0)    	
    	self.w = self.w - dEdw * lr
    	self.b = self.b - dEdb * lr
    	return
    def error(self, x, t):
        z = np.argmax(self.fprop(x), axis = 1)
        errors = np.sum(z!=t)
        return errors

# layer 1
L = layer()
# mesuring time
start = time.time()
for i in range(5):
    # training
    for j in range(50000/batch):
    	L.update(.0001, train_set_x[j*batch:(j+1)*batch], train_set_y[j*batch:(j+1)*batch])
    # mesuring error rate
    error_count = 0
    for j in range(10000/batch):
        error_count += L.error(valid_set_x[j*batch:(j+1)*batch], valid_set_y[j*batch:(j+1)*batch])
    error_rate = error_count / 10000.
    # report
    print "validation error rate : " + str(error_rate) + " epoch " + str(i) 
print " execution time : " + str(time.time() - start)
    
    

    	
