import numpy as np
# from sklearn.metrics import r2_score

class BGD:
    def __init__(self,lr=0.1,n_iter=1000,see_loss=False, tol=None):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss = []
        self.tol = tol
        self.see_loss = see_loss
        self.loss_old = 1
    def loss_func(self,y,y_pred):
        return np.mean((y-y_pred)**2)
    def gradient(self,X,y,y_pred):
        dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
        db = -2*np.mean(y-y_pred)
        return dw,db
    def predict(self,X):
        return np.dot(X,self.w.T)+self.b
    def fit(self,X,y):
        self.w = np.ones(X.shape[1])
        self.b = np.random.randn()
        for i in range(self.n_iter):
            y_pred = np.dot(X,self.w.T)+self.b
            loss = self.loss_func(y,y_pred)
            self.loss.append(loss)
            dw,db = self.gradient(X,y,y_pred)
            self.w -= self.lr*dw
            self.b -= self.lr*db
            if self.see_loss:
                if (i+1)%100 == 0:
                    print("Epoch: {}, Loss: {}".format(i+1,loss))
            if self.tol is not None:
                if (abs(self.loss_old-loss)/self.loss_old) < self.tol:
                    print("Converged at epoch: {}".format(i+1))
                    break
                self.loss_old = loss
    def score(self, X, y):
        y_pred = self.predict(X)
        r2_n = np.sum((y-y_pred)**2)
        r2_d = np.sum((y-y.mean())**2)
        return 1 - (r2_n/r2_d)


class SGD():
    def __init__(self,lr=0.1,n_iter=1000,see_loss=False,tol=None):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss = []
        self.see_loss = see_loss
        self.loss_old = 1
        self.tol = tol
    def loss_func(self,y,y_pred):
        return np.mean((y-y_pred)**2)
    def gradient(self,X,y,y_pred):
        dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
        db = -2*np.mean(y-y_pred)
        return dw,db
    def predict(self,X):
        return np.dot(X,self.w.T)+self.b
    def fit(self,X,y):
        self.w = np.ones(X.shape[1])
        self.b = np.random.randn()
        for i in range(self.n_iter):
            for _ in range(X.shape[0]):
                idx = np.random.randint(0,X.shape[0])
                X_i = X[idx]
                y_i = y[idx]
                y_pred = self.predict(X_i)
                dw,db = self.gradient(X_i,y_i,y_pred)
                self.w -= self.lr*dw
                self.b -= self.lr*db
            y_hat = self.predict(X)
            loss = self.loss_func(y,y_hat)
            self.loss.append(loss)

            if self.see_loss == True:
                if (i+1)%10 == 0:
                    print("Epoch: {}, Loss: {}".format(i+1,loss))
            if self.tol is not None:
                if (abs(self.loss_old-loss)/self.loss_old) < self.tol:
                    print("Converged at epoch: {}".format(i+1))
                    break
                self.loss_old = loss
    def score(self, X, y):
        y_pred = self.predict(X)
        r2_n = np.sum((y-y_pred)**2)
        r2_d = np.sum((y-y.mean())**2)
        return 1 - (r2_n/r2_d)

























































































































































# class BGD:
#     def __init__(self,lr=0.1,n_iter=1000,see_loss=False):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.w = None
#         self.b = None
#         self.loss = []
#         self.see_loss = see_loss
#     def loss_func(self,y,y_pred):
#         return np.mean((y-y_pred)**2)
#     def gradient(self,X,y,y_pred):
#         dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
#         db = -2*np.mean(y-y_pred)
#         return dw,db
#     def predict(self,X):
#         return np.dot(X,self.w.T)+self.b
#     def fit(self,X,y):
#         self.w = np.ones(X.shape[1])
#         self.b = np.random.randn()
#         for i in range(self.n_iter):
#             y_pred = np.dot(X,self.w.T)+self.b
#             loss = self.loss_func(y,y_pred)
#             self.loss.append(loss)
#             dw,db = self.gradient(X,y,y_pred)
#             self.w -= self.lr*dw
#             self.b -= self.lr*db
#             if self.see_loss:
#                 if (i+1)%100 == 0:
#                     print("Epoch: {}, Loss: {}".format(i+1,loss))

#     def score(self, X, y):
#         y_pred = self.predict(X)
#         r2_n = np.sum((y-y_pred)**2)
#         r2_d = np.sum((y-y.mean())**2)
#         return 1 - (r2_n/r2_d)


# class SGD():
#     def __init__(self,lr=0.1,n_iter=1000,see_loss=False):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.w = None
#         self.b = None
#         self.loss = []
#         self.see_loss = see_loss
#     def loss_func(self,y,y_pred):
#         return np.mean((y-y_pred)**2)
#     def gradient(self,X,y,y_pred):
#         dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
#         db = -2*np.mean(y-y_pred)
#         return dw,db
#     def predict(self,X):
#         return np.dot(X,self.w.T)+self.b
#     def fit(self,X,y):
#         self.w = np.ones(X.shape[1])
#         self.b = np.random.randn()
#         for i in range(self.n_iter):
#             for _ in range(X.shape[0]):
#                 idx = np.random.randint(0,X.shape[0])
#                 X_i = X[idx]
#                 y_i = y[idx]
#                 y_pred = self.predict(X_i)
#                 dw,db = self.gradient(X_i,y_i,y_pred)
#                 self.w -= self.lr*dw
#                 self.b -= self.lr*db
#             y_hat = self.predict(X)
#             loss = self.loss_func(y,y_hat)
#             self.loss.append(loss)

#             if self.see_loss == True:
#                 if (i+1)%10 == 0:
#                     print("Epoch: {}, Loss: {}".format(i+1,loss))

#     def score(self, X, y):
#         y_pred = self.predict(X)
#         r2_n = np.sum((y-y_pred)**2)
#         r2_d = np.sum((y-y.mean())**2)
#         return 1 - (r2_n/r2_d)



























































# class BGD:
#     def __init__(self,lr=0.1,n_iter=1000,see_loss=False):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.w = None
#         self.b = None
#         self.loss = []
#         self.see_loss = see_loss
#     def loss_func(self,y,y_pred):
#         return np.mean((y-y_pred)**2)
#     def gradient(self,X,y,y_pred):
#         dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
#         db = -2*np.mean(y-y_pred)
#         return dw,db
#     def fit(self,X,y):
#         self.w = np.ones(X.shape[1])
#         self.b = np.random.randn()
#         for i in range(self.n_iter):
#             y_pred = np.dot(X,self.w.T)+self.b
#             loss = self.loss_func(y,y_pred)
#             self.loss.append(loss)
#             dw,db = self.gradient(X,y,y_pred)
#             self.w -= self.lr*dw
#             self.b -= self.lr*db
#             if self.see_loss:
#                 if (i+1)%100 == 0:
#                     print("Epoch: {}, Loss: {}".format(i+1,loss))
    
#     def predict(self,X):
#         return np.dot(X,self.w.T)+self.b
#     def score(self, X, y):
#         y_pred = self.predict(X)
#         r2_n = np.sum((y-y_pred)**2)
#         r2_d = np.sum((y-y.mean())**2)
#         return 1 - (r2_n/r2_d)


    # def score(self, X, y):
    #     y_pred = self.predict(X)
    #     return r2_score(y,y_pred)



# class SGD():
#     def __init__(self,lr=0.1,n_iter=1000,see_loss=False):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.w = None
#         self.b = None
#         self.loss = []
#         self.see_loss = see_loss
#     def loss_func(self,y,y_pred):
#         return np.mean((y-y_pred)**2)
#     def gradient(self,X,y,y_pred):
#         dw = -(2*np.dot(X.T,y-y_pred))/X.shape[0]
#         db = -2*np.mean(y-y_pred)
#         return dw,db
#     def fit(self,X,y):
#         self.w = np.ones(X.shape[1])
#         self.b = np.random.randn()
#         for i in range(self.n_iter):
#             for _ in range(X.shape[0]):
#                 idx = np.random.randint(0,X.shape[0])
#                 X_i = X[idx]
#                 y_i = y[idx]
#                 y_pred = np.dot(X_i,self.w.T)+self.b
#                 dw,db = self.gradient(X_i,y_i,y_pred)
#                 self.w -= self.lr*dw
#                 self.b -= self.lr*db
#             y_hat = np.dot(X,self.w.T)+self.b
#             loss = self.loss_func(y,y_hat)
#             self.loss.append(loss)

#             if self.see_loss == True:
#                 if (i+1)%10 == 0:
#                     print("Epoch: {}, Loss: {}".format(i+1,loss))
#     def predict(self,X):
#         return np.dot(X,self.w.T)+self.b
#     def score(self, X, y):
#         y_pred = self.predict(X)
#         r2_n = np.sum((y-y_pred)**2)
#         r2_d = np.sum((y-y.mean())**2)
#         return 1 - (r2_n/r2_d)

        