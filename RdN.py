import cPickle, gzip, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set

train_set_x = np.reshape(train_set_x,(50000,1,784))
valid_set_x = np.reshape(valid_set_x,(10000,1,784))

# image = np.reshape(train_set_x[0], (28, 28))
# print np.shape(image)
# imgplot = plt.imshow(image)
# plt.imshow(image, cmap = cm.Greys_r)
# plt.show()


def one_hot(t):
    res = np.array([(t == 0), (t ==1), (t == 2), (t == 3), (t == 4), (t == 5), (t == 6), (t ==7), (t == 8), (t == 9)])
    res = np.reshape(res,(1,10))
    return res

#the RdN
class layer(object):
    def __init__(self): # Notre methode constructeur        
        self.w = np.zeros((28*28,10))
        self.b = np.zeros((1,10))
    def fprop(self, x):
    	z = np.dot(x, self.w) + self.b
    	return z
    def update(self, lr, x, t):
    	dEdb = 2*(self.fprop(x)-one_hot(t))
    	dEdw = np.dot(x.T, dEdb)    	
    	self.w = self.w - dEdw * lr
    	self.b = self.b - dEdb * lr
    	return
    def error(self, x, t):
        if t == np.argmax(self.fprop(x)):
            return 0
        else:
            return 1

L = layer()

for i in range(15):
    for j in range(50000):
    	L.update(.0001, train_set_x[j], train_set_y[j])
    error_count = 0
    for j in range(10000):
        error_count = error_count + L.error(valid_set_x[j], valid_set_y[j])
    error_rate = error_count / 10000.
    print "validation error rate : " + str(error_rate) + " epoch " + str(i)
    
    

    	
