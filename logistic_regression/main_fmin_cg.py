# this file will use only fmin_cg for calculating the optimized theta

class main:
    class logisreg:
        def __init__(self, y, *features):
            self.m = len(y)
            self.features = [np.ones(self.m), *features]
            self.X = np.asarray(self.features).T
            self.Y = np.reshape(y, (len(y), 1))
            
        def hx(self, X, theta):
            return expit(X@theta)
            
        def cost_func(self, theta, *args):
            X, y = args
            lambda_val = 1
            X = X.reshape(len(y), -1)
            hx = self.hx( X, theta)
            grad = (1/len(y))*(X.T @ (hx - y))
            hx, y = hx.ravel(), y.ravel()
            cost = (-1/len(y))* (y*np.log(hx) + (1-y)*np.log(1 - hx+1e-10) ).sum()
            regularized_cost = cost + (theta[1:]*theta[1:]).sum()*(lambda_val/len(y))
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
            print('Theta optimized: ', self.theta)
                                    
        def predict(self, x1, x2, y):
            features = [np.ones(len(x1)), x1,x2]
            prediction = self.hx(np.asarray(features).T, self.theta)
            return prediction, self.cost_func(self.theta, np.asarray(features).T, np.asarray(y.to_list()))

    def plot_decision_boundary(self, df, theta):
        #sns.lmplot('x1', 'x2', data=df, hue='y', fit_reg=False)
        """
        since hx = theta0 + theta1*x1 + theta2*x2, thus on hx, this will be zero, or x2 = -(theta0 + theta1*x1)/theta2
        """
        theta = theta.ravel()
        df['x2_modified'] = df.x1.apply(lambda x: -(theta[0] + theta[1]*x)/theta[2])
        colors = np.where(df.y == 0, 'red', 'blue')
        ax = df.plot(kind= 'scatter', x = 'x1', y ='x2', color = colors)  # this is the trick to plot on same figure.
        df.plot(x = 'x1', y = 'x2_modified', ax = ax)
        plt.show()
    
    def run(self):
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
        else:
            with open('data.csv', 'w+') as f: pass
            df = pd.DataFrame(list(zip(np.random.randint(2, 100, 50),
            np.random.randint(2, 100, 50), np.random.choice([0, 1], 50))), columns = ['x1','x2', 'y'])
            df.to_csv('data.csv', index = False)
        df_train, df_test = df.head(8* len(df)//10).copy(), (df.iloc[8* len(df)//10:]).copy()
        regressor = self.logisreg(np.asarray(df_train.y.to_list()), 
                                  *[np.asarray(df_train[i].to_list()) for i in df_train.columns[:-1]])
        regressor.fit()
        #model, hx_train = regressor.fit()
        hx, cost= regressor.predict(df_test.x1, df_test.x2, df_test.y)
        print('cost: ', cost)
        self.plot_decision_boundary(df, regressor.theta)





if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    from scipy.special import expit
    from scipy import optimize
    import matplotlib.pyplot as plt
    main().run()
