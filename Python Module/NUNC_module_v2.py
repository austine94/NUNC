# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:02 2022

@author: austine
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:48:30 2022

@author: austine
"""
from sortedcontainers import SortedList
from collections import deque
import math
import matplotlib.pyplot as plt
    
def nunc(X,k,w,threshold) : 
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    def quantiles(data,k,w) :
        #function for computing the k quantiles for data with window of size w
        def quantile(prob) : #compute single interpolated quantile
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs]    
    def update(tree,window,x) : #updates current tree and window for new point 
        if len(window) == window.maxlen :
            tree.remove(window[0])
        window.append(x)
        tree.add(x)
        return (tree,window)
    def argmax(l) :  
        pos = max(range(len(l)),key=lambda i: l[i])
        return (l[pos],pos)
    def eCDF_vals(data,quantile):    #used to return value of eCDF, not cost     
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        return val
    def one_point_emp_dist(data, quantile):
        #function for computing empirical CDF for data at a particular quantile
        #data is an array of numerics
        #quantile is the quantile to evaluate the eCDF at.
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0)
    def cdf_cost(cdf_val, seg_len):
        #function for computing the likelihood function using the value of the CDF
        #cdf_val is the value of the eCDF at a set quantile
        #seg_len is the length of the data used 
        if(cdf_val <= 0 or cdf_val >= 1):
            return(0) #avoids error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    def update_window_ecdf_removal(data_to_remove, quantiles, current_ecdf, current_len):
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= current_len
            current_ecdf[i] -= one_point_emp_dist(data_to_remove, quantiles[i]) 
            current_ecdf[i] /= (current_len - 1)  
        return current_ecdf
    tree = SortedList()
    window = deque([],w)
    dtime = 0
    cost_max_history = []
    for x in X[:(w-1)] : #first fill an initial window
        tree,window = update(tree,window,x)
        dtime += 1
    #from this point we have w points so start testing
    for x in X[(w-1):] : #each iteration is O(log(W) + WK)
        tree,window = update(tree,window,x)
        dtime += 1        
        Q = quantiles(tree,k,len(window)) #update quantiles
        full_cdf_vals = [eCDF_vals(tree, q) for q in Q] #full ecdf values
        full_cost = sum(cdf_cost(val, w) for val in full_cdf_vals) #cost for window
        right_cdf_vals = full_cdf_vals.copy() #this will be updated as we search the window for changes
        segment_costs = list()
        current_len = w
        left_cdf_vals = [0] * len(Q) #also to be updated as we search
        for i in range(0, w-1): #window updates are O(K)
        #iteratively remove points from RHS and update eCDF
            right_cdf_vals = update_window_ecdf_removal(window[i], Q, right_cdf_vals, 
                                                        current_len)
            current_len -= 1
            #we can recover cost of LHS using full eCDF and RHS
            for j in range(len(Q)):
                left_cdf_vals[j] = (full_cdf_vals[j]*w - right_cdf_vals[j]*current_len) / (w - current_len)
            #we can now compute the costs of the left and right hand sides
            left_cost = sum([cdf_cost(val, w - current_len) for val in left_cdf_vals])
            right_cost = sum([cdf_cost(val, current_len) for val in right_cdf_vals])
            segment_costs.append(left_cost + right_cost)
        #return max cost and the position in the window
        cost_max,pos = argmax(segment_costs)
        cost_max = 2*(cost_max - full_cost)
        cost_max_history.append(cost_max)
        if cost_max > threshold :
            return(dtime-w+pos,dtime,cost_max, cost_max_history)
    cost_max = max(cost_max_history)
    return (-1, -1, cost_max, cost_max_history)


########################
#NUNC Global Code
#######################

def nunc_global(X,k,w,threshold) :
    
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    
    def quantiles(data,k,w) :
    #function for computing the k quantiles for data with window of size w
        def quantile(prob) :
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs]    
    def cost(data,quantile) :
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        conj = 1 - val
        return 0 if val <= 0 or val >= 1 else len(data)*(val*math.log(val)-conj* math.log(conj))
    def update(tree,window,x) :
        if len(window) == window.maxlen :
            tree.remove(window[0])
        window.append(x)
        tree.add(x)
        return (tree,window)
    def update_window(window, x):   
        window.append(x)
        return window
    def eCDF_vals(data,quantile):    #used to return value of eCDF, not cost     
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        return val     
    def cdf_cost(cdf_val, seg_len):
        #function for computing the likelihood function using the value of the CDF
        #cdf_val is the value of the eCDF at a set quantile
        #seg_len is the length of the data used 
        if(cdf_val <= 0 or cdf_val == 1):
            return(0) #avoids error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    def one_point_emp_dist(data, quantile):
        #function for computing empirical CDF for data at a particular quantile
        #data is an array of numerics
        #quantile is the quantile to evaluate the eCDF at.
        ####Only for updating the z vals with a single point so O(1)#####
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0)
    def update_z_vals(new_data, quantiles, current_z_vals, t, w):
        #updates the z values based on the new point
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_z_vals[i] *= (t-w)   #rescale existing z values
            current_z_vals[i] += one_point_emp_dist(new_data, quantiles[i])
            current_z_vals[i] /= (t - w + 1)
        #O(K) as we only use for new_data of a single point
        return(current_z_vals)
    def update_window_ecdf(new_data, old_data, quantiles, current_ecdf, w):
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= w
            current_ecdf[i] -= one_point_emp_dist(old_data, quantiles[i]) 
            current_ecdf[i] += one_point_emp_dist(new_data, quantiles[i]) 
            current_ecdf[i] /= w  
        return current_ecdf
    #begin algorithm
    tree = SortedList()
    window = deque([],w)
    dtime = 0
    cost_history = []
    #first get initial z vals from first w points
    for x in X[:w] : #use tree to get O(log(w)) quantiles
        tree,window = update(tree,window,x)
        dtime += 1
    Q = quantiles(tree,k,len(window)) #compute fixed quantiles
    z_vals = [eCDF_vals(tree, q) for q in Q] #compute initial z_vals 
    window_ecdf_vals = z_vals.copy() #current ecdf
    #next move forward till we have a window of all new points
    for x in X[w:2*w-1]:
        window_ecdf_vals = update_window_ecdf(x, window[0], Q,
                                              window_ecdf_vals, w)
        window = update_window(window,x)
        dtime += 1
        #no need to update z vals as we pop values already in initial estimate
        #start testing from w:2w-1 as this is the (w+1)th point on the left 
    for x in X[2*w-1:] :
        dtime += 1
        #update z vals with point that is going to leave the window,
        #and then update tree and window
        window_ecdf_vals = update_window_ecdf(x, window[0], Q,
                                              window_ecdf_vals, w)
        window = update_window(window,x)
        #compute eCDFs for weighted data           
        weighted_ecdf = [((dtime - w) / dtime)*z_vals[i] + (w / dtime)*window_ecdf_vals[i]
                         for i in range(k)]
        #compute cost functions
        historic_cost = sum([cdf_cost(val, dtime - w) for val in z_vals])
        window_cost = sum([cdf_cost(val, w) for val in window_ecdf_vals])
        full_cost = sum([cdf_cost(val, dtime) for val in weighted_ecdf])

        overall_cost = 2*(historic_cost + window_cost - full_cost)
        #update z_vals with the point that is to leave window
        z_vals = update_z_vals(window[0], Q, z_vals, dtime, w)
        cost_history.append(overall_cost)
        if overall_cost > threshold :
            return(dtime-w,dtime,overall_cost, cost_history)
        #update z vals with z val leaving window
    max_cost = max(cost_history)
    return (-1, -1, max_cost, cost_history)

#######################
#Sequential Implementations
#######################


def argmax(l) :
    #this function is needed to compute the location of the max when 
    #performing each sequential update
    pos = max(range(len(l)),key=lambda i: l[i])
    return (l[pos],pos) 

def create_nunc_state(x) :
    #this creates the initial sorted list and deque (window) for use with local
    return (SortedList(x),deque(x,maxlen=len(x) + 1))
    #note we have length + 1 as we need to add the next point and start testing
    #when we use the nunc_local_update as per below

def nunc_local_update(S,x) : 
    def quantiles(data,k,w) :
        def quantile(prob) :
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs]    
    def update(tree,window,x) :
        if len(window) == window.maxlen :
            tree.remove(window[0])
        window.append(x)
        tree.add(x)
        return (tree,window)
    def eCDF_vals(data,quantile):    #used to return value of eCDF, not cost     
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        return val
    def one_point_emp_dist(data, quantile):
        #function for computing empirical CDF for data at a particular quantile
        #data is an array of numerics
        #quantile is the quantile to evaluate the eCDF at.
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0)
    def cdf_cost(cdf_val, seg_len):
        #function for computing the likelihood function using the value of the CDF
        #cdf_val is the value of the eCDF at a set quantile
        #seg_len is the length of the data used 
        if(cdf_val <= 0 or cdf_val >= 1):
            return(0) #avoids rounding error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    def update_window_ecdf_removal(data_to_remove, quantiles, current_ecdf, current_len):
        #updates ecdf when removing a single point from it
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= current_len
            current_ecdf[i] -= one_point_emp_dist(data_to_remove, quantiles[i]) 
            current_ecdf[i] /= (current_len - 1)  
        return current_ecdf
    tree,window,k = S #update state
    tree,window = update(tree,window,x) #update with new point
    w = len(window)
    Q = quantiles(tree,k,w) #update quantiles
    full_cdf_vals = [eCDF_vals(tree, q) for q in Q] #full data eCDF
    right_cdf_vals = full_cdf_vals.copy() #will update as we search for change in window
    full_cost = sum(cdf_cost(val, w) for val in full_cdf_vals) #full data cost
    segment_costs = list()
    current_len = w
    left_cdf_vals = [0] * len(Q) #again will update as we search window for points
    for i in range(0, w-1): #window updates are O(K)
        right_cdf_vals = update_window_ecdf_removal(window[i], Q, right_cdf_vals, 
                                                    current_len)
        #remove points from RHS iteratively and update eCDF
        current_len -= 1
        for j in range(len(Q)): #update LHS using RHS and full eCDFs
            left_cdf_vals[j] = (full_cdf_vals[j]*w - right_cdf_vals[j]*current_len) / (w - current_len)
        #compute costs of segmented data
        left_cost = sum([cdf_cost(val, w - current_len) for val in left_cdf_vals])
        right_cost = sum([cdf_cost(val, current_len) for val in right_cdf_vals])
        segment_costs.append(left_cost + right_cost)
    #return full costs for each location and also updated tree / window
    costs = [2*(cost - full_cost) for cost in segment_costs]
    return (tree,window,costs)

def nunc_local(X,k,w,threshold) : 
    #wrapper function for sequential nunc
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    
    tree,window = create_nunc_state(X[:w-1])
    dtime = w-1
    for x in X[w-1:] :
        tree,window,costs = nunc_local_update((tree,window,k),x)
        dtime += 1        
        cost_max,pos = argmax(costs)
        if cost_max > threshold :
            return(dtime-w+pos,dtime,cost_max)
    return None


##############
#Global
###########

def create_nunc_state_global(x) :     
    return (SortedList(x),deque(x,maxlen=len(x)))
##starts with full window so no need to add 1
#unlike the local case that initialises with a window of length w-1 so needs 
#to have room to store the next observation 

def nunc_global_update(S,x) :    
    def update_window(window,x) :
        window.append(x)
        return (window)
    def eCDF_vals(data,quantile):    #used to return value of eCDF, not cost     
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        return val   
    def cdf_cost(cdf_val, seg_len):
        if(cdf_val == 0 or cdf_val == 1):
            return(0) #avoids error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    def one_point_emp_dist(data, quantile):
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0)  
    def update_z_vals_unbounded(new_data, quantiles, current_z_vals, w):
        #updates the z values based on the new point
        #this is modified so it does not depend on t, hence suitable for
        #use sequentially in an unbounded setting
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_z_vals[i] *= (w-1)  #rescale existing z values
            current_z_vals[i] += one_point_emp_dist(new_data, quantiles[i])
            current_z_vals[i] /= w
        return(current_z_vals)
    def update_window_ecdf(new_data, old_data, quantiles, current_ecdf, w):
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= w
            current_ecdf[i] -= one_point_emp_dist(old_data, quantiles[i]) 
            current_ecdf[i] += one_point_emp_dist(new_data, quantiles[i]) 
            current_ecdf[i] /= w   
        return current_ecdf
    
    window,k,Q,z_vals,window_ecdf_vals = S
    #state contains quantiles and z_vals now as these are fixed    
    w = len(window)  #use for updating the window
    #update the z_vals
    z_vals = update_z_vals_unbounded(window[0], Q, z_vals, w)            
    #compute eCDFs for window and full data
    window_ecdf_vals = update_window_ecdf(x, window[0], Q,
                                          window_ecdf_vals, w)
    window = update_window(window,x)
    #compute eCDFs for weighted data           
    weighted_ecdf = [0.5*z_vals[i] + 0.5*window_ecdf_vals[i]
                     for i in range(k)]
    #compute cost functions
    historic_cost = sum([cdf_cost(val, w) for val in z_vals])
    window_cost = sum([cdf_cost(val, w) for val in window_ecdf_vals])
    full_cost = sum([cdf_cost(val, 2*w) for val in weighted_ecdf])

    overall_cost = 2*(historic_cost + window_cost - full_cost)
    
    return(window, z_vals, window_ecdf_vals, overall_cost)

def nunc_global_sequential(X,k,w,threshold) :
    #wrapper function for nunc global sequential
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    
    def quantiles(data,k,w) :
        def quantile(prob) :
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs] 
    def eCDF_vals(data,quantile):    #used to return value of eCDF, not cost     
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        return val
    def one_point_emp_dist(data, quantile):
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0) 
    def update_window_ecdf(new_data, old_data, quantiles, current_ecdf, w):
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= w
            current_ecdf[i] -= one_point_emp_dist(old_data, quantiles[i]) 
            current_ecdf[i] += one_point_emp_dist(new_data, quantiles[i]) 
            current_ecdf[i] /= w  
        return current_ecdf
    #initialise z values:
    tree, window = create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    #initialise eCDF using next w points then start testing
    tree,window = create_nunc_state_global(X[w:(2*w)])
    current_ecdf_vals = [eCDF_vals(tree, q) for q in Q]
    dtime = 2*w-1
    for x in X[2*w:] :
        window,z_vals,current_ecdf_vals,cost = nunc_global_update((window,k,Q,
                                                   z_vals,current_ecdf_vals),x)
        dtime += 1 
        if cost > threshold :
            return(dtime-w,dtime,cost)
    return None

############
#NUNC Offline Wrapper
############

def nunc_offline(data, k, w, threshold, method = "local"):
    
    '''
    Function for performing NUNC in an offline setting.
    
    Description
    ------------
    
    NUNC offline applies the NUNC algorithm to a pre-observed stream of data
    and searches this data for changes in distribution using a sliding window.
    
    Two different variants of NUNC exist: "NUNC Local" and "NUNC Global".
    Each of these three variants tests for changes in distribution through use
    of a cost function that makes a comparison between the pre and post change
    empirical CDFs for the data. This comparison is aggregated over K quantiles,
    as this enhances the power of the test by comparing both the centre, and
    tails, of the estimated distributions.
    
    The two different methods can be described as follows:

    NUNC Local
    This method searches for a change in distribution inside the points of
    data contained in the sliding window. An approximation for this algorithm
    can also be specified, that only searches a subset of the points in the
    sliding window for a change in order to enhance computational efficiency.

    NUNC Global
    This method tests if the data in the sliding window is drawn from a
    different distribution to the historic data.
    
    Parameters
    --------------
    
    data: list
        List of data to test using NUNC.
    threshold: float
        Threshold for the NUNC test.
    w: int
        Window size used by NUNC.
    K: int
        Number of quantiles used by NUNC. Must be less than the 
        size of the window.
    method: string
        To specify either "local" or "global" variant of NUNC.
        
    Returns
    ---------
    NUNC Object: NUNC_out
        An object of class NUNC_out containing the detection time, changepoint,
        max of the test statistics, list of test statistics for
        each window that is checked, and the data that was inputted.
    
    '''
    
    if not isinstance(method, str):
        raise TypeError("method must be a string of either local or global")
    
    if method.lower() == "local":
        (pos, dtime, cost_max, cost_history) = nunc(data, k, w, threshold)
    elif method.lower() == "global":
        (pos, dtime, cost_max, cost_history) = nunc_global(data, k, w, threshold)
    else:
        raise ValueError("method must be either local or global")
    
    res = NUNC_out(dtime, pos, cost_max, cost_history, data)
    return(res)
        
class NUNC_out:
    
    '''
    Class for storing the output of the NUNC offline algorithm.
    
    Attributes
    ------------
    changepoint: int
        The time NUNC identifies as the changepoint. -1 if no change
    detection_time: int
        The time NUNC identifies the changepoint. None if no change detected.
    cost_max: float
        The max of the costs observed - if a change is detected this will be
        the first value to exceed the threshold.
    cost_vec: list
        The list of max of NUNC test statistics for each window. 
    data: list
        The list of data inputted.
    
    Methods
    -----------
    
    plot(None):
        Method to plot the data, and (if detected the )changepoint and 
        time of detection
            
    '''
    
    def __init__(self, dtime, pos, cost_max, cost_history, data):
        self.detection_time = dtime
        self.changepoint = pos
        self.cost_max = cost_max
        self.cost_history = cost_history
        self.data = data
        
    def plot(self):
        data_length = len(self.data)
        data_length = len(self.data)
        x_points = list(range(data_length))
        plt.plot(x_points, self.data)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        if self.detection_time != -1:
            plt.axvline(x = self.changepoint, color = "red", 
                        label = "changepoint")
            plt.axvline(x = self.detection_time, color = "blue",
                        label = "Detection Time")
            plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
