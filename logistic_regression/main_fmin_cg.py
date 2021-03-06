# this file will use only fmin_cg for calculating the optimized theta
# if other data files are to be classified, make sure it is csv, has y column at last, and length is >= 10.

class main:
    class logisreg:
        def __init__(self, y, *features):
            self.m = len(y)
            self.features = [np.ones(self.m), *features] # it means, all are linear, i.e. theta0x0 + theta1x1 + ...
            self.X = np.asarray(self.features).T
            self.Y = np.reshape(y, (len(y), 1))
            
        def hx(self, X, theta):
            return expit(X@theta)
            
        def cost_func(self, theta, *args):
            X, y = args
            lambda_val = 1
            X = X.reshape(len(y), -1)
            hx = self.hx( X, theta)
            hx, y = hx.ravel(), y.ravel()
            cost = (-1/len(y))* (y*np.log(hx) + (1-y)*np.log(1 - hx+1e-10) ).sum()
            regularized_cost = cost + (theta[1:]*theta[1:]).sum()*(lambda_val/len(y))
            return cost
            
        def grad(self, theta, *args):
            """ gradient of cost function"""
            X, Y = args
            X = X.reshape(len(Y), -1)
            hx = self.hx( X, theta)
            rate = (1/len(Y))*(X.T @ (hx - Y)) # result will be a vector containing cost for each theta 
            return rate.ravel()
            
        def fit(self):
            # uses fmin_cg algorithm, a conjugate gradient algorithm, Advanced Optimization.
            self.theta = optimize.fmin_cg(self.cost_func,            #cost function
                                    np.ones(len(self.features)),     #initial theta
                                    self.grad,                       #gradient of cost function
                                    args = (self.X.ravel(), self.Y.ravel())) #extra args needed
            print('Theta optimized:', self.theta)
                                    
        def predict(self, y, *features):
            features = [np.ones(len(y)), *features]
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
    
    def run(self, datafile):
        if os.path.exists(datafile):
            df = pd.read_csv(datafile)
            """
            for polynomial regression:- make df modified in df.columns[:-1], and do modification.
            Ex: If your df has columns x1, x2, x3 ... y, then if you want it to be polynomial like
            x1**a, x2**b, x3**c etc, then:
            def foo(x, *args):
                cols = args[0]
                lc = list(cols)
                powers = [a, b, c, d, ..., 1]
                
                return x**(powers[lc.index(x.name)])
                
            df.apply(foo, args = (df.columns,))
            
            Ex: For df of 4 cols:
            def foo(x, *args):
                cols = args[0]
                lc = list(cols)
                powers = [1,2, 3, 1]
                assert len(df.columns) == len(powers), 'powers ill defined'
                try: return x**(powers[lc.index(x.name)])
                except IndexError: print(lc, powers, x.name, lc.index(x.name)); exit()
                
            df.apply(foo, args = (df.columns,))
            """
            
        else:
            print('DataFile "{}" not found, creating it based on simple sample.'.format(datafile))
            with open(datafile, 'w+') as f: pass
            df = pd.DataFrame(list(zip(np.random.randint(2, 100, 50),
            np.random.randint(2, 100, 50), np.random.choice([0, 1], 50))), columns = ['x1','x2', 'y'])
            df.to_csv(datafile, index = False)
        df_train, df_test = df.head(8* len(df)//10).copy(), (df.iloc[8* len(df)//10:]).copy()
        regressor = self.logisreg(np.asarray(df_train.y.to_list()), 
                                  *[np.asarray(df_train[i].to_list()) for i in df_train.columns[:-1]])
        regressor.fit()
        #model, hx_train = regressor.fit()
        hx, cost= regressor.predict(df_test.y, *[df_test[i] for i in df_test.columns[:-1]] )
        print('cost: ', cost)
        if len(df.columns) == 3: self.plot_decision_boundary(df, regressor.theta)





if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    from scipy.special import expit #sigmoid function
    from scipy import optimize
    import matplotlib.pyplot as plt
    main().run('data.csv')
