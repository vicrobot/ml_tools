import numpy as np
import pandas as pd
import os
from scipy import optimize
from scipy.special import expit

class main:
    class NN:
        def __init__(self, hidden_layer_count, hidden_layer_size):
            """
            Neural Network class.
            What it basically is doing:-
            1. It has self.L layers.
            2. It does forward propagation to get values of al ie layers' node value.
            3. After that, we use backpropagation to minimize our cost function.
               Cost function takes note of thetas in regularization and the logistical sort of cost by log function,
               in such a way that if |hx - y| is big, the cost would be big, and vice versa.
            4. Backpropagation involves calculation of deltas, corresponding to each node in the NN. 
            Going from right to left, to calc the val. of del_l's jth node, 
            we can sum all weights times the del it is coming from.
            It is like alloting of error's map occured at output layer to inner nodes. And when value tries to flow
            in forward direction(the same val by which we calc. dels), 
            then we multiply val of node with del value of that node. 
            That mul. is like representative of dataloss of all past data sort of thing idk even roughly clearly.
            5. After that, as usual, we can use theta:=theta-alpha*grad or any other adv. optimization func.
            """
            self.hlc = hidden_layer_count
            self.hls = hidden_layer_size
            self.L = self.hlc + 2 #layers' count
            #self.times = 1e3
            self.lambda_val = 1e0
        
        def cost_func(self, theta):
            m = self.m
            X, y = self.X, self.y
            
            origin, layer1s, layer2s = 0, (self.ils+1), (self.hls)
            layer = X.copy()
            for i in range(1, self.L): #forward prop
                layer_wb = np.asarray([np.ones(len(layer)), *layer.T]).T #layer with bias unit
                theta_curr = theta[origin: (origin + layer1s*layer2s)].reshape(layer1s,layer2s) #current layer theta
                origin = layer1s*layer2s
                layer = self.hx(layer_wb, theta_curr)
                layer1s = self.hls + 1
                layer2s = self.hls if i < self.L - 2 else self.ols
            
            cost = (-1/m)*( y*np.log(layer) + (1-y)*np.log(1-layer) ).sum()
            reg_term = (self.lambda_val/(2*m))*((theta**2).sum())
            return cost + reg_term
        
        def hx(self, X, theta):
            #assuming X has m examples, n features, + 1 bias unit,
            #assuming theta is a mat of shape (n+1, hls), +1 'cuz theres bias unit theta0
            return expit(X@(theta)) #result's shape is (m, hls)
        
        def grad(self, theta):
            X = self.X
            y = self.y
            
            origin, layer1s, layer2s = 0, (self.ils+1), (self.hls)
            layer_ls = [] #this will have all unbiased layers in increasing order.
            theta_ls = [] #this will have all thetas in increasing order from theta1 to thetaL-1.
            biased_ls = [] #this will have all biased list in increasing order ie l1, l2, ... lL-1. Not lL.
            layer = X.copy()
            layer_ls.append(layer.copy())
            for i in range(1, self.L): #forward prop
                layer_wb = np.asarray([np.ones(len(layer)), *layer.T]).T #layer with bias unit
                biased_ls.append(layer_wb.copy())
                theta_curr = theta[origin: (origin + layer1s*layer2s)].reshape(layer1s,layer2s) #current layer theta
                theta_ls.append(theta_curr.copy())
                origin = layer1s*layer2s
                layer = self.hx(layer_wb, theta_curr)
                layer_ls.append(layer.copy())
                layer1s = self.hls + 1
                layer2s = self.hls if i < self.L - 2 else self.ols
            
            layer_idx = -1
            bias_idx = -1
            del_last = layer_ls[layer_idx] - y
            del_ls = [] #this will have deltas in decreasing order due to back prop. Thus del for last layer is on 0 idx
            del_ls.append(del_last.copy())

            # del of last layer is appended. Now we've to calculate del for all layers except inp layer.
            # For that we'll do del_l = (thetal.T)*del_l+1 .*al .*(1-al)
            # where for a part. layer's del's calc., you'll have to get theta for that layer - 1 num.
            # Ex: for last - 1 layer, you need last theta.
            # Also thats why below we iterate self.L - 2 times since out is processed and inp layer has not to process
            
            for i in range(1, self.L-1): #backward prop
                #del_l = (thetal.T)*del_l+1 .*al .*(1-al)
                # al.*(1-al) is sigmoid gradient. (Have to read it though since idk'boutthat)
                if i>1:
                    del_ = (del_ls[-1][:,1:]@theta_ls[bias_idx].T)*biased_ls[bias_idx]*(1-biased_ls[bias_idx])
                else: del_ = (del_ls[-1]@theta_ls[bias_idx].T)*biased_ls[bias_idx]*(1-biased_ls[bias_idx])
                #reasons for above if-else:
                # Last layer delta has shape m*output_nodes, and no bias unit in that.
                # Also know that nodal value of layer l+1 is affected by layer l's bias unit , but not converse.
                # So consider this simple situation. If we've got l layers, l-1's bias' delta is mistake because 
                # of that bias. But since no past layers < l-1 has ever touched bias node of l-1 layer,
                # thus to calc del of past layer in back prop, we leave the bias layer.
                # One more view angle is that; we do let forces of mistake come from future layers ie  layers > l-a
                # to get mistake value of bias unit of l-a and then we improve it in theta updation. But we stop that
                # force at that bias unit and don't blame past layer's node for mistake.
                
                #we started removing bias from layer l-1, which was used for del of l-2, thus we did that else at i>1
                del_ls.append(del_.copy())
                bias_idx -= 1
            
            raveled_accum = []
            del_idx = -1
            biased_idx = 0
            for i in range(1, self.L-1): #theta updation #self.L-1 since last theta is updated seperately after loop
                Theta_grad = (1/self.m)*(del_ls[del_idx][:,1:].T@biased_ls[biased_idx]).T
                theta_grad_reg = (self.lambda_val/self.m)*(theta_ls[biased_idx][:,1:]).sum()
                Theta_grad+= theta_grad_reg
                raveled_accum.extend(Theta_grad.ravel())
                del_idx-=1
                biased_idx += 1
                
            Thetalast_grad = (1/self.m)*(del_ls[0].T@biased_ls[-1]).T #no biased unit in last layer.
            thetalast_grad_reg = (self.lambda_val/self.m)*(theta_ls[-1][:,1:]).sum()
            Thetalast_grad+= thetalast_grad_reg
            raveled_accum.extend(Thetalast_grad.ravel())
            #print(".",end="",flush=True)
            return np.asarray(raveled_accum).ravel()
            
        def fit(self, X, y):
            self.m, self.n = X.shape
            self.ils = self.n #inp_layer_size
            self.ols = len(y[0]) #out_layer_size
            self.hls = self.hls #hidden layer size
            self.X = X
            self.y = y
            #random initialization
            #init_thetas = np.random.rand(ils + 1, hls)
            #for i in range(self.L-3):
            #    init_thetas.extend(np.random.rand(hls +1, hls))
            #init_thetas += np.random.rand(hls + 1, ols)
            
            
            #optimize func needs cost func, init thetas, grad func, args. theta is given to funcs by default.
            self.Theta = optimize.fmin_cg(self.cost_func,
                                          np.random.rand((self.ils+1)*self.hls + (self.L-3)*(self.hls+1)*self.hls + 
                                          (self.hls+1)*self.ols),
                                          self.grad)
            print('Theta Optimized', self.Theta)
            
        def predict(self, X):
            theta = self.Theta
            origin, layer1s, layer2s = 0, (self.ils+1), (self.hls)
            layer = X.copy()
            for i in range(1, self.L):
                layer_wb = np.asarray([np.ones(len(layer)), *layer.T]).T #layer with bias unit
                theta_curr = theta[origin: (origin + layer1s*layer2s)].reshape(layer1s,layer2s) #current layer theta
                origin = layer1s*layer2s
                layer = self.hx(layer_wb, theta_curr)
                layer1s = self.hls + 1
                layer2s = self.hls if i < self.L - 2 else self.ols
            return layer
        
        """
        def predict1(self, X):
            Theta1=  self.Theta[:(self.n+1)*(self.hls)].reshape(self.n+1,self.hls)
            Theta2= self.Theta[(self.n+1)*(self.hls):].reshape(self.hls + 1, len(self.y[0]))
            
            X = np.asarray([np.ones(len(X)), *X.T]).T
            hx2=self.hx(X, Theta1)#shape is (m, hls)'s corresponding, m here test's examples' count
            hx2_wb = np.asarray([np.ones(len(hx2)), *hx2.T]).T
            hx3=self.hx(hx2_wb,Theta2)
            prediction = hx3
            return prediction """
    
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
        hidden_layer_count = 1
        hidden_layer_size = 25
        model = self.NN(hidden_layer_count, hidden_layer_size)
        
        #fitting the model
        model.fit(X_train, self.make_vects(y_train))
        
        #predicting the result
        print("-"*20)
        prediction_vects = model.predict(X_test)
        #prediction_vects1 = model.predict1(X_test)
        prediction = pd.Series(np.argmax(prediction_vects, axis= 1).ravel()).apply(lambda x: self.uniqs[x]).to_numpy()
        df_res = pd.DataFrame({"p":prediction, "y":y_test})
        count_success = (df_res.p == df_res.y).sum()
        count_failure = len(df_res) - count_success
        print(df_res.head())
        print(f" success: {count_success}, failure: {count_failure}, odds of win: {count_success/len(df_res)}")
        
# deciding the size of hidden layers, deciding lambda,  and such values has huge effect on training the stuff.
# Do some analysis on how you could have found optimum value of such consts in this example.
# even the sample size if affecting the model on large scale even when negligible changes are taken there like
# 9/10 -> 8/10.

if __name__ == "__main__":
    main().run("digits.csv")




