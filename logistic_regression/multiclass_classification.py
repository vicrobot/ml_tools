#Meta:
"""
let there are k objects to be classified.
Let object be x, and we're doing digit classification, then
Y X prob_that_X_is_Y(hx) 
0 x   something near to 0
1 x   something near to 0
2 x   ''
3 x   ''
4 x   ''
5 x   something near to 0.89   ---------------> This is highest, thus it classifies that x is 5
6 x   something near to 0
7 x   something near to 0.12
8 x   something near to 0.52
9 x   something near to 0.23
"""
"""
For this; Y or say label would be a vector of size n and on nth place it has 1 else 0, and X would be a matrix of m*n size.

We'll train data for 1 and non 1, 2 and non 2 etcs till 0 and non 0. Thus y will be a vector here.
"""

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
            lambda_val = 0.001
            X = X.reshape(len(y), -1)
            hx = self.hx( X, theta)
            grad = (1/len(y))*(X.T @ (hx - y))
            hx, y = hx.ravel(), y.ravel()
            cost = (-1/len(y))* (y*np.log(hx) + (1-y)*np.log(1 - hx+1e-10) ).sum()
            regularized_cost = cost + (theta[1:]*theta[1:]).sum()*(lambda_val/len(y))
            return cost
            
        def grad(self, theta, *args):
            X, Y = args
            lambda_val = 0.001
            X = X.reshape(len(Y), -1)
            hx = self.hx( X, theta)
            rate = (1/len(Y))*(X.T @ (hx - Y)) # result will be a vector containing cost for each theta 
            regularized_rate = rate + (theta[1:]).sum()*(lambda_val/len(Y))
            return regularized_rate.ravel()
            
        def fit(self):
            # uses fmin_cg algorithm, a conjugate gradient algorithm, Advanced Optimization.
            self.theta = optimize.fmin_cg(self.cost_func,
                                    np.ones(len(self.features)), 
                                    self.grad, 
                                    args = (self.X.ravel(), self.Y.ravel()))
            print('Theta optimized: ', self.theta)
                                    
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
    
    def multiclass_prediction(self, y,theta, *features):
        features = [np.ones(len(y)), *features]
        X= np.asarray(features).T
        main = expit(X @ theta.T) # gives a matrix where each row has probs for each classes for each row example of X.
        return np.argmax(main, axis = 1) # returns index of maximum probability.
    
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
            
            def foo(x, *args):
                cols = args[0]
                lc = list(cols)
                powers = list(range(1, len(lc))) + [1] # that last 1 is for y's powers
                assert len(df.columns) == len(powers), 'powers ill defined'
                try: return x**(powers[lc.index(x.name)])
                except IndexError: print(lc, powers, x.name, lc.index(x.name)); exit()
                
            df.apply(foo, args = (df.columns,))
            #df= df.head(20).copy()
            """
        else:
            print('DataFile "{}" not found, creating it based on simple sample.'.format(datafile))
            with open(datafile, 'w+') as f: pass
            df = pd.DataFrame(list(zip(np.random.randint(2, 100, 50),
            np.random.randint(2, 100, 50), np.random.choice([0, 1], 50))), columns = ['x1','x2', 'y'])
            df.to_csv(datafile, index = False)
        labels = np.asarray(list('0123456789')) #labels change based on situation, like list('0123456789') ; ['x1','x2']
        
        df= df.sample(frac =1).reset_index(drop = True)
        df_train, df_test = df.iloc[: 8* len(df)//10].copy(), df.iloc[8* len(df)//10 :].copy()
        
        #weights cooking:-------------------------
        
        if os.path.exists('theta.pickle'):
            with open('theta.pickle', 'rb+') as var:
                theta = pickle.load(var)
        else:
            theta = np.zeros((len(labels), len(df.columns)))
            for i in range(1, len(labels) + 1):
                temp_y = np.where(df_train.y == i, 1, 0)
                print(temp_y)
                monoregressor = self.logisreg(np.asarray(temp_y), #df_train.y.to_list()
                                      *[np.asarray(df_train[i].to_list()) for i in df_train.columns[:-1]])
                monoregressor.fit()
                if i == len(labels):
                    theta[0] = monoregressor.theta
                else: theta[i] = monoregressor.theta
            print('model_trained')
            with open('theta.pickle', 'wb+') as var:
                pickle.dump(theta, var)
        
        #predicting:-------------------------------
        
        prediction = labels[self.multiclass_prediction(np.asarray(df_test.y.to_list()), theta,
                             *[np.asarray(df_test[i].to_list()) for i in df_test.columns[:-1]] )]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = pd.DataFrame({'p': prediction, 'y': df_test.y}, dtype = np.int64).reset_index(drop = True)
            res.y = np.where(res.y == 10, 0, res.y)
        success = np.where(res.p == res.y, 1, 0).sum()
        failure = len(res) - success
        print('success:', success, ', failure:', failure, ', odds of favor:', success/len(res))
        
        #animating:--------------------------------
        
        #return #enable it to stop animation
        """
        class matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, 
        save_count=None, *, cache_frame_data=True, **kwargs)
        """
        fig, ax = plt.subplots()
        axes_obj = ax.imshow(np.asarray(df_test.iloc[0][:-1].to_list()).reshape(20, 20).T)
        print('Predicted: ', end = '', flush = True)
        gray = plt.cm.gray
        def init():
            img_arr = np.asarray(df_test.iloc[0][:-1].to_list()).reshape(20, 20).T
            print('\b{}'.format(res.iloc[0][0]), end = '', flush = True)
            return [ax.imshow(img_arr, cmap = gray)]
        def animate(i):
            img_arr = np.asarray(df_test.iloc[i][:-1].to_list()).reshape(20, 20).T
            print('\b{}'.format(res.iloc[i][0]), end = '', flush = True)
            return [ax.imshow(img_arr, cmap = gray)]
        
        ani = anim.FuncAnimation(fig, animate, frames = np.arange(1, len(df_test)), init_func = init, interval = 1500,
                                 blit = True)
        plt.show()
        print()
            
        


if __name__ == '__main__':
    import numpy as np
    import random
    import pandas as pd
    import os
    from scipy.special import expit
    from scipy import optimize
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import pickle
    import warnings
    #warnings.filterwarnings('error')
    main().run('digits.csv')
