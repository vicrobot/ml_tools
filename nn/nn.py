import numpy as np
import pandas as pd
import os
from scipy import optimize
from scipy.special import expit

class main:
    class NN:
        def __init__(self, hidden_unit_count, hidden_unit_size):
            """
            Neural Network class.
            What it basically is doing:-
            1. It has self.L layers.
            2. It does forward propagation to get values of al ie layers' node value.
            3. After that, we use backpropagation to minimize our cost function.
               Cost function takes note of thetas in regularization and the logistical sort of cost by log function,
               in such a way that if |hx - y| is big, the cost would be big, and vice versa.
            4. Backpropagation involves calculation of deltas, corresponding to each node in the NN. Going from right to left, to calc the val. of del_l's jth node, we can sum all weights times the del it is coming from.
            It is like alloting of error's map occured at output layer to inner nodes. And when value tries to flow
            in forward direction(the same val by which we calc. dels), then we multiply val of node with del value of that node. That mul. is like representative of dataloss of all past data sort of thing idk even roughly clearly.
            5. After that, as usual, we can use theta:=theta-alpha*grad or any other adv. optimization func.
            """
            self.huc = hidden_unit_count
            self.hus = hidden_unit_size
            self.L = self.huc + 2 #layers' count
            #self.times = 1e3
            self.lambda_val = 1e0
        
        def cost_func(self, theta):
            m = self.m
            X, y = self.X, self.y
                        
            self.Theta1= theta[:(self.n+1)*(self.hus)].reshape(self.n+1,self.hus)
            self.Theta2= theta[(self.n+1)*(self.hus):].reshape(self.hus + 1, len(self.y[0]))
            
            X_wb = np.asarray([np.ones(self.m), *X.T]).T #now its shape is m*(n+1) #wb means with bias
            hx2 = self.hx(X_wb, self.Theta1) # resultant is m*self.hus mat; containing a2s for each example
            hx2_wb = np.asarray([np.ones(len(hx2)), *hx2.T]).T # now its shape is m*(self.hus+1)
            hx3 = self.hx(hx2_wb, self.Theta2) # resultant is m*len(y[0]) means clearly containing output layers.
            
            theta_mats = theta
            cost = (-1/m)*( y*np.log(hx3) + (1-y)*np.log(1-hx3) ).sum()
            reg_term = (self.lambda_val/(2*m))*((theta_mats**2).sum())
            return cost + reg_term
        
        def hx(self, X, theta):
            #assuming X has m examples, n features, + 1 bias unit,
            #assuming theta is a mat of shape (n+1, hus), +1 'cuz theres bias unit theta0
            return expit(X@(theta)) #result's shape is (m, hus)
        
        def grad(self, theta):
            X = self.X
            y = self.y
            
            self.Theta1= theta[:(self.n+1)*(self.hus)].reshape(self.n+1,self.hus)
            self.Theta2= theta[(self.n+1)*(self.hus):].reshape(self.hus + 1, len(self.y[0]))
            
            X_wb = np.asarray([np.ones(self.m), *X.T]).T #now its shape is m*(n+1) #wb means with bias
            hx2 = self.hx(X_wb, self.Theta1) # resultant is m*self.hus mat; containing a2s for each example
            hx2_wb = np.asarray([np.ones(len(hx2)), *hx2.T]).T # now its shape is m*(self.hus+1)
            hx3 = self.hx(hx2_wb, self.Theta2) # resultant is m*len(y[0]) means clearly containing output layers.
            
            del_3 = hx3 - y #dims are m*len(y[0])
            #del_l = (thetal.T)*del_l+1 .*al .*(1-al)
            del_2 = (del_3@self.Theta2.T)*hx2_wb*(1-hx2_wb) #dim is m*self.hus+1
            #we only calculate del till del_2
            #del_1 = 0  since input doesn't have any mistake.
            
            #del of a node  * a prev layer node_val = gradient of cost func wrt theta or weight that connects the forward flow bw em
            Theta1_grad = (1/self.m)*(del_2[:,1:].T@X_wb).T # of same dim as theta1 is
            Theta2_grad = (1/self.m)*(del_3.T@hx2_wb).T #of same dim as theta2, we deleted no bias term since there is no bias unit in third layer delta(which is output layer). #divided by self.m, since they got accumulation of m examples.
            
            theta1_grad_reg = (self.lambda_val/self.m)*(self.Theta1[:,1:]).sum()
            theta2_grad_reg = (self.lambda_val/self.m)*(self.Theta2[:,1:]).sum()
            Theta1_grad+= theta1_grad_reg
            Theta2_grad+= theta2_grad_reg
            return np.asarray([*Theta1_grad.ravel(), *Theta2_grad.ravel()])
            
        def fit(self, X, y):
            self.m, self.n = X.shape
            #for now, i'll assume self.huc = 1,
            self.X = X
            self.y = y
            #random initialization
            self.Theta1 = np.random.rand(self.n + 1, self.hus)
            self.Theta2 = np.random.rand(self.hus + 1, len(y[0]))
            
            #optimize func needs cost func, init thetas, grad func, args. theta is given to funcs by default.
            self.Theta = optimize.fmin_cg(self.cost_func,
                                          np.asarray([*self.Theta1.ravel(), *self.Theta2.ravel()]),
                                          self.grad)
            print('Theta Optimized', self.Theta)
            
        def predict(self, X):
            Theta1=  self.Theta[:(self.n+1)*(self.hus)].reshape(self.n+1,self.hus)
            Theta2= self.Theta[(self.n+1)*(self.hus):].reshape(self.hus + 1, len(self.y[0]))
            
            X = np.asarray([np.ones(len(X)), *X.T]).T
            hx2=self.hx(X, Theta1)#shape is (m, hus)'s corresponding, m here test's examples' count
            hx2_wb = np.asarray([np.ones(len(hx2)), *hx2.T]).T
            hx3=self.hx(hx2_wb,Theta2)
            prediction = hx3
            return prediction
    
    
    def make_vects(self, label_list):
        """
        Takes in the classification label_list, and assigns a 0 1 vector to it.
        Ex: If its a binary classification with label like anything like A and B, then
        vectors alloted or mapped for A is [1, 0] and for B is [0, 1]
        """
        self.uniqs = list(set(label_list))
        l_uniq = len(self.uniqs)
        l_ll = len(label_list)
        vects = np.zeros((l_ll, l_uniq))
        for i in range(l_ll):
            vects[i][self.uniqs.index(label_list[i])] = 1
        return np.asarray(vects)
        
        
    def run(self, input_file):
        #checking if input file exists
        if not os.path.exists(input_file): raise FileNotFoundError(f"file {input_file} not found"); exit()
        
        #reading file data and making test and train splits
        df = pd.read_csv(input_file)
        df = df.sample(frac = 1).reset_index(drop=True)
        df_train, df_test = df.iloc[: 9* len(df)//10].copy(), df.iloc[9* len(df)//10 :].copy()
        X_train, y_train = df_train[df_train.columns[:-1]].to_numpy(), df_train[df_train.columns[-1]].to_numpy()
        X_test, y_test = df_test[df_test.columns[:-1]].to_numpy(), df_test[df_test.columns[-1]].to_numpy()
        
        #specifying parameters and making neural network object.
        hidden_unit_count = 1 
        hidden_unit_size = 25
        model = self.NN(hidden_unit_count, hidden_unit_size)
        
        #fitting the model
        model.fit(X_train, self.make_vects(y_train))
        
        #predicting the result
        prediction_vects = model.predict(X_test)
        prediction = pd.Series(np.argmax(prediction_vects, axis= 1).ravel()).apply(lambda x: self.uniqs[x]).to_numpy()
        df_res = pd.DataFrame({"p":prediction, "y":y_test})
        count_success = (df_res.p == df_res.y).sum()
        count_failure = len(df_res) - count_success
        print(df_res.head())
        print(f" success: {count_success}, failure: {count_failure}, odds of win: {count_success/len(df_res)}")
        
# deciding the size of hidden units, deciding lambda,  and such values has huge effect on training the stuff.
# Do some analysis on how you could have found optimum value of such consts in this example.


if __name__ == "__main__":
    main().run("digits.csv")

