#!/usr/bin/env python3
"""
Created on Tue Jun  9 18:16:23 2020

@author: Arnaud Gardille
"""

import numpy as np

#import pickle
#import gzip
import matplotlib.pyplot as plt
from time import time
from numpy.random import normal
from scipy.stats import norm
#from scipy.spatial.distance import euclidean
from numpy.linalg import svd
import matplotlib.animation as animation

def sigmoid(x):
    #pasDeNan(x)
    x = bornage(x)
    res = 1/(1+np.exp(-x))
    return res


def reduc(x, V):
    return np.dot(x, V)
    #[0:2]
    #return np.dot(x, V).T[:2].T

def getMiniBatches(X,m,bs):
    return X[m*bs:(m+1)*bs]

def bornage(x): # A SUPPRIMER
    return x
    """
    l = 10

    x = np.maximum(x, -l)
    x = np.minimum(x, l)

    return x
    """

def isNan(x):
    return np.any(x != x)

def pasDeNan(x):
    if np.any(x != x):
        raise RuntimeError
        
def estimVarianceClusters(rbm, Center, nb = 1000):
    Ng = Center.shape[0]
    
    
    PtCluster = [[] for i in range(Ng)]
    N = np.zeros(Ng, dtype=int)

    # Genearting data from the RBM
    v = 20*np.random.random((nb, rbm.num_visible)) - 10
    X = rbm.gibbsSampling(v, 100)
    
    
    # computing distance from the centers for each point
    D = [np.mean(np.absolute(X - Center[k]), axis=1) for k in range(Ng)]
    
    Nearest = np.argmin(D, axis=0)
    N = [np.count_nonzero(Nearest == k) for k in range(Ng)]
    print(N)
    
    for i in range(nb):
        PtCluster[Nearest[i]].append(X[i])
    #print(n)
    
    print(np.array(PtCluster[0]).shape)
    
    s = [np.sum(np.array(PtCluster[i])**2) for i in range(Ng)]
    #
    print()
    print(s)
    
    return [np.sqrt(s[i]/N[i])*(1 + 0.25*(1-np.exp(-rbm.num_visible/3))) for i in range(Ng)]
    """
    if n != 0:
        return np.sqrt(s/n)*(1 + 0.25*(1-np.exp(-rbm.num_visible/3)))
    else:
        return 0
        
    """

def genNuage(rbm, P, k=100, nbEpoch = 100, nb = 100, premCo = False):
        #nb = 100
        X = []
        Y = []
        #print("debut")
        Xcoo = []
        Ycoo = []

        for i in range(nb):
            #print(i)
            v = 20*np.random.random(rbm.num_visible) - 10
            v = rbm.gibbsSampling(v, k)
            x, y = reduc(v, P)
            #x = v[0]
            #y = v[1]

            X.append(x)
            Y.append(y)

            x = v[0]
            y = v[1]

            Xcoo.append(x)
            Ycoo.append(y)

            #if i%20 == 0:
            #    print("*", end='')

        print()
        plt.title("100 points générées après %s epochs d'entrainement" % int(nbEpoch))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.plot(X, Y, 'xb', label='selon la PCA')
        if premCo:
            plt.plot(Xcoo, Ycoo, 'xr', label='premières coordonnées')
        plt.legend()
        plt.savefig("%s epochs" % int(nbEpoch))
        plt.show()

"""
def genNuage3D(rbm, P, k=100, nbEpoch = 100, nb = 100):
    
        #nb = 100
        #X = []
        #Y = []
        #Z = []

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(nb):
            #print(i)
            v = 20*np.random.random(rbm.num_visible) - 10
            v = rbm.gibbsSampling(v, k)
            x, y, z = v[:3]

            #X.append(x)
            #Y.append(y)
            #Z.append(z)

            ax.scatter(x, y, z)


            if i%20 == 0:
                print("*", end='')

        print()
        plt.title("100 points générées après %s epochs d'entrainement" % int(nbEpoch))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.savefig("%s points générés après %s epochs d'entrainement" % (nb, int(nbEpoch)))
        plt.show()
"""



def affData(data, V):
        """Les 1000 premiers points"""
        X = np.dot(data, V)[:,0]
        Y = np.dot(data, V)[:,1]



        #X = data[:,0]
        #Y = data[:,1]
        #print(X.shape)

        plt.title("Données (2 premières composantes)")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.plot(X, Y, 'xb', label="selon la PCA")
        X = data[:,0]
        Y = data[:,1]
        #plt.plot(X, Y, 'xr', label="réelles")
        plt.legend()
        plt.savefig("Données")
        plt.show()

def affData3D(data, V):
        """Les 1000 premiers points"""
        X = data[:1000,0]
        Y = data[:1000,1]
        Z = data[:1000,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for x, y, z in zip(X, Y, Z) :
            ax.scatter(x, y, z)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.savefig("Données3D")
        plt.show()

class GBRBM_recuite():
    def __init__(self, num_visible, num_hidden, P = np.zeros(1), moy = np.zeros(1), Center = np.zeros(1), mean = 0, Ng = 0):

        if np.all(P == np.zeros(1)):
            0
        if np.all(moy == np.zeros(1)):
            0
        if np.all(Center == np.zeros(1)):
            0
        if np.all(mean == np.zeros(1)):
            0
        if np.all(Ng == np.zeros(1)):
            0
        
        self.printEvolParam = True
        self.computeVariance = False
        self.saveRBM = True
        

        self.mean = mean

        self.P = P
        self.moy = moy
        #self.eqType = eqType
        self.Ng = Ng
        self.lr = None #set during the learning

        self.xCenter = np.zeros(self.Ng)
        self.yCenter = np.zeros(self.Ng)
        for i in range(self.Ng):
            redu = reduc(Center[i], self.P)
            self.xCenter[i] = redu[0]
            self.yCenter[i] = redu[1]

        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = False
        self.k = 30 # number of Gibbs sameling steps

        self.hidden_bias = np.zeros(num_hidden)
        self.visible_bias = np.zeros(num_visible)


        self.L_errors = []
        self.L_b = []
        self.L_c = []
        self.L_lr = []
        self.L_sigma = []

        #self.cote = int(self.num_visible ** (1/2))

        self.printNuage = True
        self.recuit = True

        #print("xCenter")
        #print(self.xCenter.shape)

        np_rng = np.random.RandomState(1234)

        #self.visible_bias = np.zeros(num_visible) # TEST


        self.weights1 = 10* np.asarray(np_rng.uniform(
    			low=- 0.1*np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))


        self.weights2 = 0.1* np.asarray(np_rng.uniform(
    			low=- 0.1 *np.sqrt(6. / (num_hidden + num_visible)),
                           	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                           	size=(num_hidden, num_visible)))


        #self.weights1 = 10*np.random.random((num_hidden, num_visible))
        """
        self.weights1 = 0.01 * np.ones((num_hidden, num_visible))

        self.weights2 = 0.01 * np.ones((num_hidden, num_visible))
        """
        self.L_W1 = []
        self.L_W2 = []
        
        self.L_W1_norm = []
        self.L_W2_norm = []

        #self.L_sigma = []

        self.Variances = []

        #if dataCluster != np.zeros() :
        self.ratioIdeal = 1/self.Ng

        #self.sigma = np.ones(self.num_visible)


    def probaHiddens(self, X, beta=1):
        #return sigmoid( beta*( np.dot(self.weights1, X) + self.hidden_bias + np.dot(self.weights2, X**2) ) )
        d1 = np.dot(self.weights1*beta, X.T).T
        d2 = np.dot(self.weights2*beta, X.T**2).T
        return sigmoid( d1 + self.hidden_bias + d2 )


    def SampleHiddens(self, X):
        p_h = self.probaHiddens(X)
        h = (np.random.random(self.num_hidden) < p_h)*1
        pasDeNan(h)
        return h
    """
    def probaVisible(self, H, beta=1):
        D = 1 - 2 * np.dot(H, self.weights2)
        return (np.dot(H, self.weights1) + self.visible_bias) / D
    """

    def SampleVisibles(self, H, beta=1):
        D = 1 - 2 * np.dot(H, beta*self.weights2) 
        #D = np.abs( 1 - 2 * np.dot(H, self.weights2) )
        #D = np.abs( 1 - 2 * self.sigma**2 * np.dot(H, self.weights2) )
        moy = (beta * np.dot(H, self.weights1) + self.visible_bias) / D
        #eqT = self.sigma / np.sqrt(D)
        eqT = np.absolute(D)**(-1/2)
        return bornage(normal(moy, np.sqrt(beta) * eqT))

    def gibbsSampling(self, v, k=100):
        for _ in range(k):
            h = self.SampleHiddens(v)
            v = self.SampleVisibles(h)
            v = bornage(v)
        return v

    def gibbsSamplingAnnealing(self, v, B_init, k=100):
        B = np.linspace(B_init, 1, int(k))
        for beta in B:
            h = self.SampleHiddens(v)
            v = self.SampleVisibles(h, beta)
            v = bornage(v)
        return v

    def train(self, data, max_epochs, lr_init):
        
        nbMoyEx = data.shape[0] / self.Ng

        self.L_W1 = list(self.L_W1)
        self.L_W2 = list(self.L_W2)
        self.L_b = list(self.L_b)
        self.L_c = list(self.L_c)


        t0 = time()
        #x_tilde = np.random.random(self.num_visible) # AMELIORABLE
        #x_tilde = 5 * np.random.random(self.num_visible)- self.mean # AMELIORABLE
        x_tilde = np.random.normal(0, 4, self.num_visible)
        self.lr = lr_init


        for epoch in range(max_epochs):

            u, s, vh = svd(self.weights1)
            s_star = max(s)
            #k = max(100, int(epoch))
            #k = max(10, int(epoch)/10)
            k=10
            if self.recuit and s_star > 1/4 :
                B_init = 1 / (4*s_star)
                x_tilde = self.gibbsSamplingAnnealing(x_tilde, B_init, k)
            else :
                x_tilde = self.gibbsSampling(x_tilde, k)
            pb = 0
            if not isNan(x_tilde):
                self.updateParameters(data, x_tilde, epoch)
            else:
                print("NAN !")
                print(x_tilde)
                x_tilde = np.random.normal(0, 4, self.num_visible)


            if self.printEvolParam :
                self.param_save()

            if self.computeVariance  and epoch % 10 == 0 :
                V = []
                for i in range(self.Ng):
                    #print(self.estimVarianceCluster(i).shape)
                    V.append(self.estimVarianceCluster(i))
                #print("V")
                print(V)
                if self.printEvolParam :
                    self.Variances.append(V)

            if self.printNuage and epoch % 10 == 0:
                #genNuage(self, self.P, k=100, nbEpoch = epoch, nb=1000)
                self.genNuageCouleur(k=100, nbEpoch = epoch, nb=1000)

            if self.saveRBM and epoch % 10 == 0 :
                np.save("Saved/%s_W1" % epoch, self.weights1)
                np.save("Saved/%s_W2" % epoch, self.weights2)
                np.save("Saved/%s_vb" % epoch, self.visible_bias)
                np.save("Saved/%s_hb" % epoch, self.hidden_bias)


        print("durée d'entraiement : %s" % (time() - t0))
    
    
    def trainBatch(self, data, max_epochs, lr, bs=10):

        self.lr = lr
        t0 = time()

        permanentChain = np.random.normal(0, 4, (bs, self.num_visible))
        
        
        for epoch in range(max_epochs):
            recuit = 0
            for m in range(int(data.shape[0]/bs)):
                mini_batch = getMiniBatches(data, m, bs)
            
                u, s, vh = svd(self.weights1)
                s_star = max(s)
                    
                if self.recuit and s_star > 1/4 :
                    B_init = 1 / (4*s_star)
                    permanentChain = self.gibbsSamplingAnnealing(permanentChain, B_init, self.k)
                else :
                    permanentChain = self.gibbsSampling(permanentChain, self.k)
    
                if not isNan(permanentChain):
                    #i = np.random.randint(permanentChain.shape[0])
                    #x_tilde = permanentChain[i]
                    """
                    x_tilde = np.mean(permanentChain, axis=0)
                    example = np.mean(mini_batch, axis=0)
                    self.updateParameters(example, x_tilde, epoch)
                    """
                    #---
                    
                    self.updateParameters(mini_batch, permanentChain, epoch)
                    
                    
                   
                else:
                    print("NAN !")
                    permanentChain = np.random.normal(0, 4, (bs, self.num_visible))

            if self.debug_print:
                annealing_rate = (100 * recuit / data.shape[0])
                print("Epoch %s: annealing %s" % (epoch,  annealing_rate))

            if self.printEvolParam :
                self.param_save()

            if self.computeVariance  and epoch % 10 == 0 :
                V = []
                for i in range(self.Ng):
                    #print(self.estimVarianceCluster(i).shape)
                    V.append(self.estimVarianceCluster(i))
                #print("V")
                print(V)
                if self.printEvolParam :
                    self.Variances.append(V)

            if self.printNuage and epoch % 10 == 0:
                #genNuage(self, self.P, k=100, nbEpoch = epoch, nb=1000)
                self.genNuageCouleur(k=100, nbEpoch = epoch, nb=1000)

            if self.saveRBM and epoch % 10 == 0 :
                np.save("Saved/%s_W1" % epoch, self.weights1)
                np.save("Saved/%s_W2" % epoch, self.weights2)
                np.save("Saved/%s_vb" % epoch, self.visible_bias)
                np.save("Saved/%s_hb" % epoch, self.hidden_bias)

        print("durée d'entraiement : %s" % (time() - t0))
        
    

    def load(self, epoch) :
        self.weights1 = np.load("Saved/%s_W1.npy" % epoch)
        self.weights2 = np.load("Saved/%s_W2.npy" % epoch)
        self.visible_bias = np.load("Saved/%s_vb.npy" % epoch)
        self.hidden_bias = np.load("Saved/%s_hb.npy" % epoch)

    def genImMoy(self, nb=100):
        ims = []
        for i in range(nb):
            v = np.random.random(self.num_visible)
            v = self.gibbsSampling(v, 1000)
            ims.append(v.reshape(self.cote, self.cote))

        im = np.sum(np.array(ims), axis=0)/100
        plt.title("moyenne de 100 images")
        plt.imshow(im)
        plt.show()

    def updateParameters(self, example, x_tilde, epoch=0):
        
        h_example = self.probaHiddens(example)
        h_tilde = self.probaHiddens(x_tilde)

        lr = self.lr
        size_minibatch = example.shape[0]
        
        """
        print()
        print(example.T.shape)
        print(h_example.shape)
        print(self.weights1.shape)
        """
        
        self.hidden_bias += lr * np.mean(h_example - h_tilde, axis=0)
        self.visible_bias += lr * np.mean(example - x_tilde, axis=0)
        
        self.weights1 += lr * ( np.dot(h_example.T, example) - np.dot(h_tilde.T, x_tilde))/size_minibatch
        self.weights2 += lr * (np.dot(h_example.T, example**2) - np.dot(h_tilde.T, x_tilde**2))/size_minibatch
        
        """
        self.hidden_bias += lr * (h_example - h_tilde)
        self.visible_bias += lr * (example - x_tilde)

        self.hidden_bias = bornage(self.hidden_bias)
        self.visible_bias = bornage(self.visible_bias)

        self.weights1 += lr * (np.outer(h_example, example) - np.outer(h_tilde, x_tilde))
        self.weights2 += lr * (np.outer(h_example, (example**2)) - np.outer(h_tilde, (x_tilde**2)))
        """




    def estimVarianceCluster(self, i, nb = 1000):
        #nb = 100
        X = []
        n = 0

        for _ in range(nb):
            #print(i)
            v = 20*np.random.random(self.num_visible) - 10
            x = self.gibbsSampling(v, 100)

            #D = [euclidean(x, Center[k]) for k in range(self.Ng)]
            D = [np.mean(np.absolute(x - Center[k])) for k in range(self.Ng)]
            if D.index(min(D)) == i:
                n += 1
                X.append(D[i])

            #Gen.append(v.copy())

            #if i%1000 == 0:
            #    print("*", end='')
        #print()

        X = np.array(X)
        #print(n)
        s = np.sum(X**2, axis=0)
        #print(s)
        if n != 0:
            return np.sqrt(s/n)*(1 + 0.25*(1-np.exp(-self.num_visible/3)))
            #return np.sqrt(s/n)
        else:
            return 0
        #return np.sqrt((1/nb)*np.sum((Gen - self.moy), axis=0))

    def estimVariance(self, nb = 10000):
        #nb = 100
        """
        X = [[] for _ in range(self.Ng)]
        N = np.zeros(self.Ng, dtype=int)
        Var = np.zeros(self.Ng)

        for _ in range(nb):
            #print(i)
            v = 20*np.random.random(self.num_visible) - 10
            x = self.gibbsSampling(v, 1000)

            #D = [euclidean(x, Center[k]) for k in range(self.Ng)]
            D = [np.mean(np.absolute(x - Center[k])) for k in range(self.Ng)]
            i = D.index(min(D))
            N[i] += 1
            X[i].append(x)

            #Gen.append(v.copy())

            #if i%1000 == 0:
            #    print("*", end='')
        #print()

        X = np.array(X)
        #print(n)
        s = np.sum(X**2, axis=0)
        #print(s)
        if N[i] != 0:
            return np.sqrt(s/N[i])
        else:
            return 0
        #return np.sqrt((1/nb)*np.sum((Gen - self.moy), axis=0))
        """

    def ajustNbPtCluster(self, dataCluster, nbMoyEx, nb = 1000):
        N = np.ones(self.Ng) * 1.0 #Help stabilising the system
        #but we vould use 0 instead
        for _ in range(nb):
            #print(i)
            v = 20*np.random.random(self.num_visible) - 10
            x = self.gibbsSampling(v, 100)
            D = [np.mean(np.absolute(x - Center[k])) for k in range(self.Ng)]
            i = D.index(min(D))
            N[i] += 1.0

        if np.any(N == 0):
            return np.concatenate(dataCluster)

        force = self.ratioIdeal * nb / N
        force /= np.max(force)

        newData = []
        rep= np.zeros(self.Ng, dtype=int)
        for i in range(self.Ng):
            rep[i]= int(force[i]*nbMoyEx)
            newData.append(dataCluster[i][:rep[i]])
        newData = np.concatenate(newData)

        if self.debug_print or True:
            print("nb")
            print(N)
            print("rep")
            print(rep)
        #print("newData")
        #print(newData.shape)
        return newData

    def genNuageCouleur(self, k=100, nbEpoch = 100, nb = 100):
        color = ['xb', 'xr', 'xg', 'xy', 'x']

        plt.title("100 points générées après %s epochs d'entrainement" % int(nbEpoch))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        Pts = [[] for _ in range(self.Ng)]

        for j in range(nb):
            #print(i)
            v = 20*np.random.random(self.num_visible) - 10
            x = self.gibbsSampling(v, k)
            D = [np.mean(np.absolute(x - Center[k])) for k in range(self.Ng)]
            i = D.index(min(D))

            X_pca = reduc(x, self.P)

            plt.plot(X_pca[0], X_pca[1], color[i])

            #if j%20 == 0:
            #   print("*", end='')

        plt.show()

    def param_save(self):
        self.L_W1_norm.append(np.linalg.norm(self.weights1))
        self.L_W2_norm.append(np.linalg.norm(self.weights2))
        #self.L_W1.append(self.W1)
        #self.L_W2.append(self.W2)
        
        self.L_b.append(np.copy(self.hidden_bias))
        self.L_c.append(np.copy(self.visible_bias))
        #self.L_sigma.append(np.copy(self.sigma))

    def affParamm(self):
        print("c : max %s, min %s" % (max(self.visible_bias), min(self.visible_bias)))
        print("b : max %s, min %s" % (max(self.hidden_bias), min(self.hidden_bias)))

        plt.title("W_1")
        plt.ylabel("hidden neurons")
        plt.xlabel("visible neurons")
        plt.imshow(self.weights1)
        plt.savefig("W_1")
        plt.show()

        plt.title("Norm of the weight matrix")
        plt.ylabel("hidden neurons")
        plt.xlabel("Epoch of training")
        self.L_W1_norm = np.array(self.L_W1_norm)
        self.L_W2_norm = np.array(self.L_W2_norm)
        absci = np.arange(self.L_W1_norm.shape[0])
        plt.plot(absci, self.L_W1_norm, label="W1")
        plt.plot(absci, self.L_W2_norm, label="W2")
        plt.legend()
        plt.savefig("Norm")
        plt.show()

        plt.title("W_2")
        plt.ylabel("hidden neurons")
        plt.xlabel("visible neurons")
        plt.imshow(self.weights2)
        plt.savefig("W_2")
        plt.show()


        plt.title("hidden bias")
        plt.ylabel("hidden neurons")
        plt.xlabel("Epoch of training")
        plt.imshow(np.array(self.L_b).T)
        plt.savefig("HB")
        plt.show()

        plt.title("visible bias")
        plt.ylabel("visible neurons")
        plt.xlabel("Epoch of training")
        plt.imshow(np.array(self.L_c).T)
        plt.savefig("VB")
        plt.show()

        """
        self.L_W1 = np.array(self.L_W1)
        self.L_W2 = np.array(self.L_W2)
        fig = plt.figure()
        ims = []
        for W in self.L_W1:
            im = plt.imshow(W, animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save('W_1.gif')
        fig = plt.figure()
        ims = []
        for W in self.L_W2:
            im = plt.imshow(W, animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save('W_2.gif')
        """

        if self.computeVariance:
            plt.title("Variances")
            plt.ylabel("Variance")
            plt.xlabel("50th of Epoch of training")
            self.Variances = np.array(self.Variances)
            absci = np.arange(self.Variances.shape[0])
            for i in range(self.Ng):
                plt.plot(absci, self.Variances[:,i], label="Cluster %s" % i)
            plt.legend()
            plt.savefig("Variances")
            plt.show()

        np.save("1000W1", self.L_W1)
        np.save("1000W2", self.L_W2)
        np.save("1000b", self.L_b)
        np.save("1000c", self.L_c)

if __name__ == '__main__':

    print("matrice de passage")
    V = np.load('V.npy')
    print(V.shape)

    print("Données")
    X = np.load('data.npy')
    print(X.shape)

    print("Moyenne")
    moy = np.load('mean.npy')
    print(moy.shape)
    
    """
    print("Equart type")
    eqType = np.load('eqType.npy')
    print(eqType.shape)
    """

    print("Centres")
    Center = np.load('center.npy')
    print(Center.shape)

    print("dataCluster")
    dataCluster = np.load('dataCluster.npy')
    print(dataCluster.shape)

    (nb, size) = X.shape
    print()
    print(size)
    print()
    affData(X, V)

    nbEpoch = 1000
    nbHidden = 30
    # dim * cluster ?
    #lr = 1e-4 / size
    lr = 1e-6
    
    
    print("GBRBM_Annealing")
    #V = np.eye(size)

    print("lr = %s" % lr)
    print("nbEpoch = %s" % nbEpoch)
    print("nbHidden = %s" % nbHidden)

    
    r1 = GBRBM_recuite(size, nbHidden, V, moy, Center, 0, 3)
    #r1.load(60)
    r1.trainBatch(X, nbEpoch, lr)
    r1.affParamm()
    #genNuage(r1, V, nbEpoch = 1000, nb=1000)

    #r1.train(X, 2000, 1e-6, 1e-6, 1000)

    #np.linalg.norm(r1.L_c, axis=0)
    #"""
    """
    r2 = GBRBM_recuite(size, nbHidden, V, moy, Center, 0, 3)
    #r1.load(60)
    r2.train(X, nbEpoch, 1e-6, dataCluster, 10e-5, ajusteData=False)
    r2.affParamm()
    """
    #"""
    """
    # lr decrease
    print("R1")
    r1 = GBRBM_recuite(size, nbHidden, V, moy, eqType, Center)
    r1.train(X, nbEpoch, 1e-5, 1e-6)
    r1.affParamm()
    genNuage(r1, V, nbEpoch = 1000, nb=1000)

     # lr cst
    print("R2")
    r2 = GBRBM_recuite(size, nbHidden, V, moy, eqType, Center)
    r2.train(X, nbEpoch, 1e-6, 1e-6)
    r2.affParamm()
    genNuage(r2, V, nbEpoch = 1000, nb=1000)

    # lr increase
    print("R3")
    r3 = GBRBM_recuite(size, nbHidden, V, moy, eqType, Center)
    r3.train(X, nbEpoch, 1e-6, 1e-5)
    r3.affParamm()
    genNuage(r3, V, nbEpoch = 1000, nb=1000)

    # lr cst
    print("R4")
    r4 = GBRBM_recuite(size, nbHidden, V, moy, eqType, Center)
    r4.train(X, nbEpoch, 1e-5, 1e-5)
    r4.printError()
    r4.affParamm()
    genNuage(r4, V, nbEpoch = 1000, nb=1000)
    """

