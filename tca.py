import numpy as np
import scipy.linalg
import sklearn.metrics

def kernel(ker,X1,X2,gamma):
    K=None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None :
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray (X1).T,np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf' :
        if X2 is not None :
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T,np.asarray(X2).T,gamma)
        else :
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T,None,gamma)
    return K

class TCA:
    def __init__(self,kernel_type='primal',dim=10, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        
    def fit(self,Xs,Xt):
        X = np.hstack ((Xs.T, Xt.T) )
        X /= np.linalg.norm(X, axis=0)
        m,n = X.shape
        ns,nt = len(Xs),len(Xt)
        e = np.vstack((1/ns*np.ones((ns,1)),-1/nt*np.ones((nt,1)))) 
        M = e*e.T
        M = M/np.linalg.norm(M,'fro')
        H = np.eye(n)-1/n*np.ones((n,n))
        K = kernel(self.kernel_type,X,None,gamma=self.gamma)
        n_eye =m if self.kernel_type == 'primal' else n
        
        a,b = np.linalg.multi_dot([K,M,K.T])+self.lamb*np.eye(n_eye),np.linalg.multi_dot([K,H,K.T])
        w,V = scipy.linalg.eig(a,b)
        ind = np.argsort(w)
        A = V[:,ind[:self.dim]]
        Z = np.dot(A.T,K)
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new,Xt_new = Z[:,:ns].T,Z[:,ns:].T
        Xs_new=np.float64(Xs_new.real)
        Xt_new=np.float64(Xt_new.real)
        return Xs_new,Xt_new