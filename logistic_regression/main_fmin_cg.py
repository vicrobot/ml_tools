# this file will use only fmin_cg for calculating the optimized theta

class main:
    class logisreg:
        def __init__(self, x1, x2, y):
            self.features = [np.ones(len(x1)), x1, x2]
            self.m = len(y)
            self.X = np.asarray(self.features).T
            self.Y = np.reshape(y, (len(y), 1))
            
        def hx(self, X, theta):
            return expit(X@theta)
            
        def cost_func(self, theta, *args):
            X, y = args
            X = X.reshape(len(y), -1)
            hx = self.hx( X, theta)
            grad = (1/len(y))*(X.T @ (hx - y))
            hx, y = hx.ravel(), y.ravel()
            cost = (-1/len(y))* (y*np.log(hx) + (1-y)*np.log(1 - hx+1e-10) ).sum()
            return cost
            
        def grad(self, theta, *args):
            X, Y = args
            X = X.reshape(len(Y), -1)
            hx = self.hx( X, theta)
            rate = (1/len(Y))*(X.T @ (hx - Y)) # result will be a vector containing cost for each theta 
            return rate.ravel()
            
        def fit(self):
            self.theta = optimize.fmin_cg(self.cost_func,
                                    np.ones(len(self.features)), 
                                    self.grad, 
                                    args = (self.X.ravel(), self.Y.ravel()))
                                    
        def predict(self, x1, x2, y):
            features = [np.ones(len(x1)), x1,x2]
            prediction = self.hx(np.asarray(features).T, self.theta)
            return prediction, self.cost_func(self.theta, np.asarray(features).T, np.asarray(y.to_list()))

    def run(self):
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
        else:
            with open('data.csv', 'w+') as f: pass
            df = pd.DataFrame(list(zip(np.random.randint(2, 100, 50),
            np.random.randint(2, 100, 50), np.random.choice([0, 1], 50))), columns = ['x1','x2', 'y'])
            df.to_csv('data.csv', index = False)
        df_train, df_test = df.head(8* len(df)//10).copy(), (df.iloc[8* len(df)//10:]).copy()
        regressor = self.logisreg(np.asarray(df_train.x1.to_list()), np.asarray(df_train.x2.to_list()),
                                                                     np.asarray(df_train.y.to_list()))
        regressor.fit()
        #model, hx_train = regressor.fit()
        hx, cost= regressor.predict(df_test.x1, df_test.x2, df_test.y)
        print('cost: ', cost)
        df_test['hx'] = np.where(hx >=0.5, 1, 0)
        fig, ax = plt.subplots(2,2, sharey=True)
        ax[0, 0].scatter(df_test[df_test.hx == 1].x1, df_test[df_test.hx == 1].x2)
        ax[1, 0].scatter(df_test[df_test.hx == 0].x1, df_test[df_test.hx == 0].x2)
        #ax[1, 0].scatter(np.arange(len(df_test.x2)), df_test.x2)
        ax[0, 1].scatter(df_test[df_test.y == 1].x1, df_test[df_test.y == 1].x2)
        ax[1, 1].scatter(df_test[df_test.y == 0].x1, df_test[df_test.y == 0].x2)
        for i in ax.ravel(): i.grid('True')
        plt.show()





if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    from scipy.special import expit
    from scipy import optimize
    import matplotlib.pyplot as plt
    main().run()
