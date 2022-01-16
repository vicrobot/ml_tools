class reg():
    def __init__(self,y, X):
        #y is output linear array of size m
        #X is matrix containing m rows, n columns, n denoting number of thetas.
        #X contains features' vals in columns.
        self.m, self.n = X.shape
        self.outp = y.reshape(1,len(y.flatten()))
        self.features = X

    def fit(self):
        #To Do: thinking on transpose of XTX's ease. I most prolly think that it can become easy.
        self.theta = self.outp@self.features@np.linalg.inv(
                        np.transpose(np.transpose(self.features)@self.features)
                        )
    
    def predict(self):
        return self.theta@np.transpose(self.features)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    m=100
    x=np.arange(m)
    n=3
    X = np.ones((m,n))
    for i in range(m):
	    for j in range(n):
		    X[i][j]=(x[i])**j  #fj(xi)
    """
    Here above, i made features from a core feature x. 
    It might be that some independent forces are making the outp. So in that case, manufacture features's vals
    manually and not by f(x) thing as they're independent thus not relating to one core force x by flavor changing f
    """
    
    l,u=20,300
    y=(np.sin(x)).reshape(1,m)
    
    model = reg(y,X)
    model.fit()
    
    plt.plot(x,y.flatten())
    plt.plot(x, model.predict().flatten())
    plt.show()
































