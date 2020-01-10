class main:
    class logisreg:
        def __init__(self, x1, x2, y):
            self.features = [np.ones(len(x1)), x1, x2]
            self.m = len(y)
            self.X = np.asarray(self.features).T
            self.Y = np.reshape(y, (len(y), 1))
            self.alpha = 5e-2 # varies according to need, also features varies too.
            self.times = 10000
            
        def hx(self, X, theta):
            #hx = theta0 + theta1 * x + theta2 * x**2
            # x's dim = m*features, theta's dim = features*1
            """
            if x is positive we are simply using 1 / (1 + np.exp(-x)) but 
            when x is negative we are using the function np.exp(x) / (1 + np.exp(x)) 
            instead of using 1 / (1 + np.exp(-x)) because when x is negative -x will be positive 
            so np.exp(-x) can explode due to large value of -x
            """
            #i am using scipy.special.expit here. It is vectorized function.
            return expit(X@theta)  #matrix mul; or use x.dot(theta)
            
        def cost_func(self, hx, y):
            """
            cost_func = (-1/m)*sum(y*log(hx) + (1-y)*log(1-hx) )
            """
            hx = hx.ravel()
            y = y.ravel()
            return (-1/len(y))* (y*np.log(hx) + (1-y)*np.log(1 - hx+1e-10) ).sum()
            
        def rate_cost_func(self, theta, X,Y):
            hx = self.hx( X, theta)
            rate = (1/len(Y))*(X.T @ (hx - Y)) # result will be a vector containing cost for each theta 
            return rate

        def thetaU(self, theta, alpha, rate_cf):
            # uses gradient descent,
            return theta - alpha*rate_cf
        def fit(self):
            # this function uses alpha, theta, X, Y, and gives us updated theta, by grad desc algorithm.
            # similar to this, fmin_cg takes in cost function, grad func, and theta, X, Y and gives us updated theta
            """
            Since hx = theta0 * x**0 + theta1 * x**1 + ...
            and theta:= theta - alpha*rate_cost_func
            and cost_func = (-1/m)*sum(y*log(hx) + (i-y)*log(1-hx) )
            or rate_cost_func =  (1/len(x))*(((hx - y).*x).sum())
            # rate_cost_func of this is same as that of linear regression.
            """
            self.theta = np.ones(len(self.features))
            self.theta = np.reshape(self.theta, (len(self.theta), 1)) # .T doesn't work in numpy for 1d arrays.
            for i in range(self.times):
                self.theta = self.thetaU(self.theta, 
                                         self.alpha, 
                                         self.rate_cost_func(self.theta, self.X, self.Y))
                cost = self.cost_func(self.hx(self.X, self.theta),self.Y)
                if self.times - i <= 100:print(cost)
            print('theta: ', self.theta.ravel())
            return self, self.hx(self.X, self.theta)
        def predict(self, x1, x2, y):
            features = [np.ones(len(x1)), x1,x2]
            prediction = self.hx(np.asarray(features).T, self.theta)
            return prediction, self.cost_func(prediction, np.asarray(y.to_list()))

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
        model, hx_train = regressor.fit()
        hx, cost= model.predict(df_test.x1, df_test.x2, df_test.y)
        print('cost: ', cost)
        df_test['hx'] = np.where(hx >=0.5, 1, 0)
        #cost = -(1/len(y))*(y*np.log(hx) + (1-y)*np.log(1- hx)).sum()
        print(df_train.head(), '\n' , df_test)
        
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
    import matplotlib.pyplot as plt
    main().run()
