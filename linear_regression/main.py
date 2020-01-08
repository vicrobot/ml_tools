class main:
    class linreg:
        def __init__(self, x, y):
            self.X = np.asarray([np.ones(len(x)), x, x**0.5]).T
            self.Y = np.reshape(y, (len(y), 1))
            self.alpha = 0.0001
            self.times = 300
        def hx(self, X, theta):
            #hx = theta0 + theta1 * x + theta2 * x**2
            # x's dim = m*features, theta's dim = features*1
            return X @ theta  #matrix mul; or use x.dot(theta)
        def rate_cost_func(self, hx, X, Y):
            return (1/np.size(X, axis = 0))*(X.T @ (hx - Y))  # result will be a vector containing cost for each theta
            
        def thetaU(self, theta, alpha, rate_cf):
            return theta - alpha*rate_cf
            
        def predict(self):
            """
            Since hx = theta0 * x**0 + theta1 * x**1 + ...
            and theta:= theta - alpha*rate_cost_func
            and cost_func = (1/(2*len(x)))* ((hx - y)**2).sum()
            or rate_cost_func =  (1/len(x))*(((hx - y).*x).sum())
            """
            self.theta = np.array([1,2, 0.4])
            self.theta = np.reshape(self.theta, (len(self.theta), 1)) # .T doesn't work in numpy for 1d arrays.
            for i in range(self.times):
                self.theta = self.thetaU(self.theta, 
                                         self.alpha, 
                                         self.rate_cost_func(self.hx(self.X, self.theta), self.X, self.Y))
            print('theta: ', self.theta.ravel())
            return self.hx(self.X, self.theta)

    def run(self):
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
        else:
            with open('data.csv', 'w+') as f: pass
            x = np.arange(100)
            df = pd.DataFrame({'x': x, 'y': (np.log(x + 1)**0.324) + 2/34})
            df.to_csv('data.csv', index = False)
        regressor = self.linreg(np.asarray(df.x.to_list()), np.asarray(df.y.to_list()))
        hx = regressor.predict()
        df  = pd.DataFrame({'x': df.x.to_list(), 'y': df.y.to_list(), 'hx': hx.ravel()})
        df.plot(x = 'x', y = ['y', 'hx'])
        plt.show()
    
            
            



if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    main().run()
