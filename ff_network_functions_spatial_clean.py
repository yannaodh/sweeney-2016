
# coding: utf-8

# In[1]:

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

try:
    get_ipython().magic(u'matplotlib inline')
except:
    pass

N_inputs = 10
N_outputs = 1

r_0 = 1.0
r_max = 20.0

dt = .001

alpha = 0.01
BCM_lambda = 0.9

theta_BCM = 1.0

theta_BCM = np.ones(N_outputs)*1.0

BCM_target = 2.0
theta_BCM_dt = .001

y_dt = 1e-2
HIP_target = BCM_target
HIP_dt = 1e-4

W_max = 1.0

pruned_synapses = False

def generate_spatial_coordinates(N_E,spatial_dist='uniform',spatial_res=0.01,diffusion_profile='gaussian',diffusion_width=0.25,periodic=False):
    """
    Generates random coordinates between (0,1) for neuron with given distribution
    Returns these coordinates, also pairwise effective diffusion distances
    """
    if spatial_dist == 'uniform':
        x_coords = np.random.uniform(0,1.0,(N_E,1))
        y_coords = np.random.uniform(0,1.0,(N_E,1))
    elif spatial_dist == 'uniform_ordered':
        x_coords = np.random.uniform(0,1.0,(N_E,1))
        y_coords = np.random.uniform(0,1.0,(N_E,1))
        x_coords.sort(0)
        y_coords.sort(0)
    elif spatial_dist == 'uniform_exc_ordered':
        x_coords= np.random.uniform(0,1.0,(N_E,1))
        y_coords= np.random.uniform(0,1.0,(N_E,1))
        x_coords.sort(0)
        y_coords.sort(0)
    elif spatial_dist == 'uniform_exc_ordered_1D':
        x_coords= np.random.uniform(0,1.0,(N_E,1))
        x_coords.sort(0)
        y_coords = x_coords.copy()
    elif spatial_dist == 'uniform_exc_ordered_nonrandom_1D':
        x_coords= np.linspace(0,1.0,N_E,endpoint=False).reshape(N_E,1)
        x_coords.sort(0)
        y_coords = x_coords.copy()
    elif spatial_dist == 'exc_1D_given':
        x_coords= exc_positions_1D_given.reshape(N_E,1)
        x_coords.sort(0)
        y_coords = x_coords.copy()
    elif spatial_dist == 'uniform_exc_2groups':
        x_coords= np.append(np.random.uniform(0,.5,(N_E/2,1)),np.random.uniform(0.5,1.0,(N_E/2,1)))
        y_coords= np.append(np.random.uniform(0,.5,(N_E/2,1)),np.random.uniform(0.5,1.0,(N_E/2,1)))
    else:
        raise ValueError("%r is not a valid spatial_dist option"% (spatial_dist))

    euclid_distances = get_spatial_euclidean_distance(x_coords,y_coords,periodic)
    diffusive_distances = get_effective_diffusion_distance(x_coords,y_coords,diffusion_profile,diffusion_width,periodic)

    return x_coords,y_coords,euclid_distances,diffusive_distances

def get_spatial_euclidean_distance(x_coords,y_coords,periodic=False):
    """
    Return euclidian distance between two neurons, given coordinates
    """
    import itertools
    assert x_coords.size == y_coords.size

    itr = itertools.permutations(range(x_coords.size),2)

    distances = np.zeros((x_coords.size,x_coords.size))

    if periodic:
        for (i,j) in itr:
            x_del = min(abs(x_coords[i]-x_coords[j]),abs(x_coords[i]+1.0-x_coords[j]),abs(x_coords[j]+1.0-x_coords[i]))
            y_del = min(abs(y_coords[i]-y_coords[j]),abs(y_coords[i]+1.0-y_coords[j]),abs(y_coords[j]+1.0-y_coords[i]))

            distances[i,j] = np.sqrt(np.square(x_del)+np.square(y_del))
    else:
        for (i,j) in itr:
            distances[i,j] = np.sqrt(np.square(x_coords[i]-x_coords[j])+np.square(y_coords[i]-y_coords[j]))

    return distances

def get_effective_diffusion_distance(x_coords,y_coords,spatial_profile='gaussian',width=0.25,periodic=False):
    """
    Return effective diffusive distance, for spatial averaging of activity
    """
    import itertools
    assert x_coords.size == y_coords.size

    itr = itertools.permutations(range(x_coords.size),2)

    distances = get_spatial_euclidean_distance(x_coords,y_coords,periodic)
    diffusion_distances = np.zeros((x_coords.size,x_coords.size))

    if spatial_profile == 'gaussian':
        for (i,j) in itr:
            diffusion_distances[i,j] = stats.norm.pdf(distances[i,j],0.0,width)
        # do diagonals as well since max pdf != 1
        for i in xrange(x_coords.size):
            diffusion_distances[i,i] = stats.norm.pdf(distances[i,i],0.0,width)
    else:
        raise ValueError("%r is not a valid spatial_dist option"% (spatial_dist))

    return diffusion_distances/np.max(diffusion_distances)

def update_rates(x):
    x[x<=0] = r_0*np.tanh(x[x<=0]/r_0)
    x[x>0] = (r_max-r_0)*np.tanh(x[x>0]/(r_max-r_0))

    return x

def update_weights(_x,_y,_W,_theta_BCM,spatial_in_weight_update=False,diffusion_distances=None,spatial_avg_type='both'):
    if spatial_in_weight_update:
        _y_spatial = np.dot(diffusion_distances,_y)/np.sum(diffusion_distances,axis=0).reshape(N_outputs,1)
        _W += alpha*_x.transpose()*_y*(_y_spatial-_theta_BCM.reshape(N_outputs,1))
        if spatial_avg_type== 'inner':
            _W += alpha*_x.transpose()*_y*(_y_spatial-_theta_BCM.reshape(N_outputs,1))
        elif spatial_avg_type== 'outer':
            _W += alpha*_x.transpose()*_y_spatial*(_y -_theta_BCM.reshape(N_outputs,1))
        elif spatial_avg_type== 'both':
            _W += alpha*_x.transpose()*_y_spatial*(_y_spatial-_theta_BCM.reshape(N_outputs,1))
        else:
            raise ValueError("%r is not a valid spatial averaging option"% (spatial_avg_type))
    else:
        _W += alpha*_x.transpose()*_y*(_y-_theta_BCM.reshape(N_outputs,1))

    # bounding weights to be positive
    _W = _W*(0.5 * (np.sign(_W) + 1))

    # bounding weights below max value
    _W[_W>W_max] = W_max

    # no self-connections
    # np.fill_diagonal(_W,0.0)

    # or, have weight decay
    _W = _W*BCM_lambda

    # set pruned synapses to zero again
    if pruned_synapses:
        _W[W_pruned==1] = 0

    return _W


def update_theta_BCM(y,theta_BCM,spatial=False,diffusion_distances=None):
    """
    This is the function which implements sliding threshold
    updates according to recent activity

    Spatial element: first will use simple spatial averaging
    with e.g. gaussian profile when updating theta_BCM
    So, no dynamics in diffusion (it's instantaneous!)
    """
    if not spatial:
        theta_BCM += theta_BCM_dt*((y/BCM_target)*y - theta_BCM)
    else:
        y_spatial_average = np.dot(diffusion_distances,y)/np.sum(diffusion_distances,axis=0).reshape(N_outputs,1)
        theta_BCM += theta_BCM_dt*((y_spatial_average/BCM_target)*y_spatial_average - theta_BCM)

    return theta_BCM

def update_state_sliding_threshold(x,y,W,H,theta_BCM,spatial=False,diffusion_distances=None,spatial_in_weight_update=False,spatial_avg_type='both'):

    x += dt*(-1*x + H)
    y += dt*(-1*y + update_rates(np.dot(W,x)))

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))
    y = y*(0.5 * (np.sign(y) + 1))

    W = update_weights(x,y,W,theta_BCM,spatial_in_weight_update,diffusion_distances,spatial_avg_type)

    theta_BCM = update_theta_BCM(y,theta_BCM,spatial,diffusion_distances)

    return x,y,W,theta_BCM

def update_state_no_plasticity(x,W,H):
    H.resize(H.size,1)
    x.resize(x.size,1)

    x += dt*(-1*x + np.dot(W,update_rates(x)) + H)

    # only allow positive firing rates
    x = x*(0.5 * (np.sign(x) + 1))

    return x.reshape(x.size,1)

def prep_net_run(T):
    from NeuroTools import stgen

    if ext_OU_noise:
        stgen = stgen.StGen()
        ext_OU = np.zeros((N_inputs,T))
        for n_idx in xrange(N_inputs):
            ext_OU[n_idx] = stgen.OU_generator_weave1(1,ext_OU_tau,ext_OU_sigma,0,0,T)[0]
        ext_OU = np.transpose(ext_OU)

    return ext_OU

def input_generator(T,N_E,N_I,H_min,H_max,input_type='constant',t_change=500,noise_type=None,OU_drive_tau=10.0,OU_drive_sigma=0.0):
    _H = np.ones((T,N_E+N_I))*H_min
    _orientations = np.zeros((T,N_orientations))

    OU_drive = np.zeros(T)
    if noise_type=='OU_drive':
        from NeuroTools import stgen

        stgen = stgen.StGen()
        ext_OU = np.zeros((N_E+N_I,T))
        OU_drive = stgen.OU_generator_weave1(1,OU_drive_tau,OU_drive_sigma,0,0,T)[0]

    if input_type == 'random_static':
    #    print 'random static input'
        _H = np.random.uniform(H_min,H_max,(N_E+N_I,1))
        _H = np.ones((1,T))*_H
        _H = _H.transpose()
    if input_type == 'given_static':
    #    print 'random static input'
        _H = H_given
        _H = np.ones((1,T))*_H
        _H = _H.transpose()

    for i in xrange(T):
        if input_type == 'random_dynamic':
            # randomly chosen subset of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_inputs)*H_min
            if i%t_change*2 == 0:
                _H[i][np.random.randint(0,N_inputs)]=H_max
            elif not i%t_change == 0:
                _H[i] = _H[i-1]
        if input_type == 'random_dynamic_leave_one':
            # randomly chosen subset of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_inputs)*H_min
            if i%t_change*2 == 0:
                _H[i][np.random.randint(1,N_inputs)]=H_max
            elif not i%t_change == 0:
                _H[i] = _H[i-1]
        elif input_type == 'oriented_dynamic':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=1
        elif input_type == 'oriented_dynamic_given_strengths':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_orientation_strengths[orientation_i]
            _orientations[i][orientation_i]=1
        elif input_type == 'oriented_dynamic_half_groups':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations/2)
                _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            if i%t_change == 0:
                _H[i][np.random.randint(N_E/2,N_E,N_E/4)]=H_max
            elif not i%t_change == 0:
                _H[i] = _H[i-1]
            _orientations[i][orientation_i]=1
        elif input_type == 'oriented_dynamic_half_groups_pairs':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations/2)
                orientation_j=orientation_i+N_orientations/2
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _H[i][orientation_j*(N_E/N_orientations):(orientation_j*(N_E/N_orientations)+N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=1
            _orientations[i][orientation_j]=1
        elif input_type == 'oriented_dynamic_spaced':
            # randomly chosen orientation of neurons receieve high input
            if i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
            _H[i][orientation_i::(N_E/N_orientations)]=H_max
            _orientations[i][orientation_i]=orientation_i
        elif input_type == 'oriented_dynamic_alternate_double':
            # randomly chosen orientation of neurons receieve high input
            if i%(t_change*2) == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_j=np.random.randint(N_orientations/2)
            elif i%t_change == 0:
                _H[i] = np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                orientation_j = None
            if not orientation_j == None:
                _H[i][orientation_j*(N_E/(N_orientations/2)):(orientation_j*(N_E/(N_orientations/2))+N_E/(N_orientations/2))]=H_max
                _orientations[i][orientation_j]=1
            else:
                _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
                _orientations[i][orientation_i]=1
        elif input_type == 'orientation_and_random_dynamic':
            if i%t_change == 0:
                _H[i]=np.random.uniform(H_min,H_max,N_E+N_I)
                orientation_i=np.random.randint(N_orientations)
                #print 'orientation_i = ', orientation_i
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]=H_max
        elif input_type == 'orientation_and_secondary_dynamic':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                secondary_i = np.random.randint(N_secondary_groups)
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max
            _H[i][secondary_groups == secondary_i] += H_secondary
        elif input_type == 'orientation_and_variable_secondary_dynamic':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                secondary_i = np.random.randint(N_secondary_groups)
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]
            _H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]
        elif input_type == 'orientation_and_secondary_dynamic_paired':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                if secondary_paired_idx[orientation_i] == None:
                    secondary_i = np.random.randint(N_secondary_groups)
                else:
                    secondary_i = secondary_paired_idx[orientation_i]
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max
            _H[i][secondary_groups == secondary_i] += H_secondary
        elif input_type == 'orientation_and_variable_secondary_dynamic_paired':
            if i%t_change == 0:
                _H[i]=np.ones(N_E+N_I)*H_min
                orientation_i=np.random.randint(N_orientations)
                if secondary_paired_idx[orientation_i] == None:
                    secondary_i = np.random.randint(N_secondary_groups)
                else:
                    secondary_i = secondary_paired_idx[orientation_i]
            _H[i][orientation_i*(N_E/N_orientations):(orientation_i*(N_E/N_orientations)+N_E/N_orientations)]+=H_max_variable[orientation_i]
            _H[i][secondary_groups == secondary_i] += H_secondary_variable[secondary_i]

        _H[i] = _H[i]+OU_drive[i]

    return _H, _orientations


def run_net_static(x,W,T=1000,N_sample=N_outputs):
    pop_rate = []

    ext_OU = prep_net_run(T)

    sample_rates = np.ones((T,N_sample))

    for i in xrange(T):
        H_noisy = H+ext_OU[i]+OU_drive[i]

        x = update_state_no_plasticity(x,W,H_noisy)

        pop_rate.append(np.mean(x))
        sample_rates[i]=x.reshape(x.size,)[:N_sample]

    return x,pop_rate,sample_rates

def run_net_plastic_sliding_threshold(x,y,W,theta_BCM,T=1000,N_sample=N_outputs,input_type='random',N_orientations=8,input_OU_sigma=0,spatial=False,diffusion_distances=None,spatial_in_weight_update=False,spatial_avg_type='both'):
    pop_rate = []

    sample_rates = np.ones((T/sample_res,N_sample))
    sample_weights = np.ones((T/sample_res,N_outputs,N_inputs))
    mean_incoming_weight = np.ones((T/sample_res,N_sample))
    sample_theta_BCM = np.ones((T/sample_res,N_sample))

    for j in xrange(T):
        if j%T_input_gen == 0:
            H,orientations = input_generator(T_input_gen,N_inputs,0,H_min,H_max,input_type,500,'OU_drive',10.0,input_OU_sigma)
            ext_OU = prep_net_run(T_input_gen)

        H.resize(H.shape[0],H.shape[1],1)
        ext_OU.resize(ext_OU.shape[0],ext_OU.shape[1],1)

        i = j%T_input_gen

        H_noisy = H[i]+ext_OU[i]

        x,y,W,theta_BCM = update_state_sliding_threshold(x,y,W,H_noisy,theta_BCM,spatial,diffusion_distances,spatial_in_weight_update,spatial_avg_type)

        if j%sample_res == 0:
            pop_rate.append(np.mean(y))
            sample_rates[j/sample_res]=y.reshape(y.size,)[:N_sample]
            mean_incoming_weight[j/sample_res]=np.sum(W,axis=1)[:N_sample]

            sample_weights[j/sample_res] = W
            sample_theta_BCM[j/sample_res] =theta_BCM.reshape(theta_BCM.size,)[:N_sample]

    return x,y,W,pop_rate,sample_rates,sample_weights,mean_incoming_weight,theta_BCM,sample_theta_BCM

