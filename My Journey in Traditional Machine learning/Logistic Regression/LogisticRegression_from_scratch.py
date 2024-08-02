import numpy as np


def sigmoid(x):    
    return 1 / (1+ np.exp(-x))
    
class LogisticRegression():
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self,X, y):
        n_samples , n_feature = X.shape 
        self.weights= np.zeros(n_feature)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            """ the loss function used is minimizing the binary cross entropy which is equavalent to maximizing likelihood """    

            dw = (1/n_samples)*(np.dot(X.T,(predictions - y)))  # dl_dw = X.T * (p - y) , derivative of binary cross entropy w.r.t to w 
            db = (1/n_samples)*np.sum(predictions-y)
            
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
    

        
    def predict(self,X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred] 
        return class_pred
    
# Testing 
# Create a random dataset and test it
np.random.seed(42)
X = np.random.randn(100,2)
t_w = np.array([0.5,-0.3])
lc = np.dot(X , t_w)
y = (lc > 0).astype(int)
model = LogisticRegression()
model.fit(X,y)
prediction = model.predict(X)
print("prediction ", prediction)
print("true labels", y.tolist())