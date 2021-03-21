import ezyrb
from ezyrb import POD, GPR, RBF, Database
from ezyrb import ReducedOrderModel as ROM
import numpy as np
from scipy.interpolation import Rbf
import cpickle  # serialization of object for python


def parameterLoading(paramLoading):
    if paramLoading:
        params=np.loadtxt("fullparams.csv", fmt='%.4f', delimiter=",")
    else:
        num_sample =19
        mu2=np.linspace(0.01,1, 19)
        num_time=161
        time= np.linspace(0,num_time,num_time)
        params= np.zeros((len(time),2) )
        params[:,0]= np.ones((len(time), 1)).reshape(-1,)
        params[:,0]*=np.round(mu2[0],4)
        params[:,1] = time
        for i in range(1, len(mu2)):
            tmp_ = np.zeros((len(time),2))
            tmp_[:,0]= np.ones((len(time), 1)).reshape(-1,)
            tmp_[:,0]*=np.round(mu2[i],4) # tmp_[:,0]= tmp_[:,0]*mu2[i]    
            tmp_[:,1] = time
            params = np.vstack((params,tmp_))
    return params

# RBF: input (parame,snapshot) 
# we choose the RBF interpolation among the parameters
for prefixSnapshot in ["U1","U2","U3"]:
    rbf = RBF(kernel='multiquadric', smooth=0.65) #smooth is ur r.
    pod = POD('svd') # rank=0.9999
    # Offline phase
    POD_restart = True # if true is loading modes and singular value
    offline = True 
    params=parameterLoading(True)
    snapshots = np.loadtxt('snapshotmatrix_{}.csv'.format(prefixSnapshot),delimiter=",",fmt='%.6e')
    if offline:    
        #POD computations and saving.
        _=pod.reduce(snapshots)
        #num_modes=pod._modes.shape[1] #numpy array shape => return tuple of 2 element, array.shape[1] num  of columns
        s = pod._singular_values
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank=np.where(cumulative_energy<0.9999999)[0]
        for i in  rank:
            np.savetxt("POD_{}_{}.csv".format(i,prefixSnapshot),pod._modes[:,i])
            #print(pod._singular_values)
        np.savetxt("sigmas_{}.csv".format(prefixSnapshot),pod._singular_values)
        np.savetxt("sigmas_selected_{}.csv".format(prefixSnapshot),pod._singular_values[rank])
        pod._modes= pod._modes[:,rank]
        # RBF here 
        db = Database(params, snapshots.T) 
        reducedcoord = pod._modes.T.dot(db.snapshots.T).T # Projection of snapshot on POD basis.
        # number of snap x number of basis
        rbf.fit(db.parameters,reducedcoord)
       
        rbfi_offline = rbf.interpolators
        RBFfile = open('picklefile_{}'.format(prefixSnapshot),'wb')
        RBFpickler = pickle.Pickler(RBFfile,protocol=2)
        # RBF can't be pickled directly, so save everything required for reconstruction
        RBFdict = {}            
        for key in rbfi_offline.__dict__.keys():
            if key != '_function' and key!= 'norm':
                RBFdict[key] = rbfi_offline.__getattribute__(key)   
        RBFpickler.dump(RBFdict)
        RBFfile.close()
    else:
        pod._singular_values=np.loadtxt("sigmas_selected_{}.csv".format(prefixSnapshot))
        modes=np.zeros((snapshots.shape[0],len(pod._singular_values)))
        for i in range(0,len(pod._singular_values)):
            modes[:,i]=np.loadtxt("POD_{}_{}.csv".format(i,prefixSnapshot))
        pod._modes = modes   
        # Load RBF
        # i need to make a bare bones arbitrally.. to instantiate it 
        rbfi_online = Rbf(np.array([1,2]), np.array([10,20]), \
                      np.array([1,2]), function = RBFdict['function'] )
        RBFfile = open('picklefile_{}'.format(prefixSnapshot),'rb')
        RBFunpickler = pickle.Unpickler(RBFfile)
        RBFdict = RBFunpickler.load()
        RBFfile.close()
        ## replace rbfi's contents with what was saved ##
        for key,value in list(RBFdict.items()): ##Python3 suppress iteritems
            rbfi_online.__setattr__(key, value)
            
        rbf.interpolators = rbfi_online
        #db = Database(params, snapshots.T) 
        #reducedcoord = pod._modes.T.dot(db.snapshots.T).T # Projection of snapshot on POD basis.
        # number of snap x number of basis
        #rbf.fit(db.parameters,reducedcoord)
        
        ##save it rbf.interpolators 

        ##ONLINE
        # Save RBF function..    
        #.....
        fem_full= np.loadtxt('{}_2.csv'.format(prefixSnapshot),delimiter=",") #mu2[1], timeserie (true) 

        error_pod_vs_fea= np.zeros((len(time)-1,1))
        for item,t in enumerate(time):
            
            new_mu = [np.round(mu2[1],4), t]
            rbf_prediction = np.atleast_2d(rbf.predict(new_mu)).T
            pod_solutionOnline=pod.expand(rbf_prediction) #prediciton with ROM-RBF
            true_solution= fem_full[:,item] #abaqus sol
            if item>0:
                error= np.linalg.norm(pod_solutionOnline-true_solution)/np.linalg.norm(true_solution)
                error_pod_vs_fea[item-1]=error
                #print("error relative l2, {} at time {} with param {}".format(error,t,new_mu[0]))    
        import matplotlib
        import matplotlib.pyplot as plt

        # Data for plotting
        fig, ax = plt.subplots()
        ax.semilogy(error_pod_vs_fea)
        fig.savefig("error_in_time{}.png".format(prefixSnapshot))
        print(error_pod_vs_fea.max())
