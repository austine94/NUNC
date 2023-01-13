# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:17:14 2022

@author: austine
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:03:04 2022

@author: austine
"""

import NUNC_DG as NUNC
import pytest
import numpy as np
import math

def test_nunc_global_update_nochange1():
    
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
    
    np.random.seed(1)
    
    w = 20
    k = 1
    X = np.random.normal(size = w + 1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[19] == X[20]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == -2.2984459377701882)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.0475)
    assert(costs ==  0.0012164887611394448)
    assert(isinstance(costs, float) )

def test_nunc_global_update_nochange2():
    
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
    
    np.random.seed(2)
    
    w = 36
    k = 4
    X = np.random.normal(size = w + 1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    
    assert(window[33] == X[34]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[1] == -2.0012851344398803)
    assert(len(z_vals) == k)
    assert(z_vals[1] == 0.05401234567901234)
    assert(costs ==   -0.050495861320843716)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_nochange3():
    
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
    
    np.random.seed(3)
    
    w = 3
    k = 2
    X = np.random.normal(size = w + 1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    
    assert(window[0] == X[1]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == 0.1648810331066879)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.2222222222222222)
    assert(costs ==   0.2982233366381035)
    assert(isinstance(costs, float) )

def test_nunc_global_update_nochange4():
    
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
    
    np.random.seed(4)
    
    w = 30
    k = 28
    X = np.random.poisson(1, size = w + 1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[0] == X[1]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == 0.0)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.11277777777777778)
    assert(costs ==   -5.883838605483199)
    assert(isinstance(costs, float) )

def test_nunc_global_update_nochange5():
    
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
    
    np.random.seed(5)
    
    w = 40
    k = 10
    X = np.random.poisson(1, size = w + 1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[w-1] == X[w]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[k-1] == 3.0)
    assert(len(z_vals) == k)
    assert(z_vals[k-1] == 0.9512499999999999)
    assert(costs ==   0.10091453018382168)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_nochange6():
    
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
    
    np.random.seed(6)
    
    w = 10
    k = 1
    X = np.repeat(3, w+1)
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[1] == X[2]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[k-1] == 3.0)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.5)
    assert(costs ==   0.0)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_change1():
    
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
    
    np.random.seed(1)
    
    w = 20
    k = 1
    X = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5,5,11)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[0] == X[1]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[k-1] == -5.262278643805498)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.0475)
    assert(costs ==   0.0012164887611394448)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_change2():
    
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
    
    np.random.seed(2)
    
    w = 40
    k = 3
    X = np.concatenate([np.random.normal(-4, 0.01, 20), np.random.normal(11,2,21)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
    
    assert(window[0] == X[1]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[1] == -4.012410147014689)
    assert(len(z_vals) == k)
    assert(z_vals[1] == 0.073125)
    assert(costs ==   0.002589369748459802)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_change3():
    
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
    
    np.random.seed(3)
    
    w = 3
    k = 2
    X = np.concatenate([np.random.normal(10, 1, 1), np.random.normal(4,1,3)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
      
    assert(window[2] == X[3]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[1] == 4.340334066091992)
    assert(len(z_vals) == k)
    assert(z_vals[1] == 0.2222222222222222)
    assert(costs ==   0.2982233366381035)
    assert(isinstance(costs, float) )
    
    
def test_nunc_global_update_change4():
    
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
    
    np.random.seed(4)
    
    w = 20
    k = 5
    X = np.concatenate([np.random.binomial(10, 0.3, 4), np.random.normal(4,0.9,17)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
     
    assert(window[w-1] == X[w]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == 2.6652376236684785)
    assert(len(z_vals) == k)
    assert(z_vals[1] == 0.0475)
    assert(costs ==   -0.04536762443716924)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_change5():
    
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
    
    np.random.seed(5)
    
    w = 30
    k = 2
    X = np.concatenate([np.random.binomial(5, 0.8, 29), np.random.normal(90,0.9,2)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
     
    assert(window[w-1] == X[w]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == 2.0)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.03222222222222222)
    assert(costs ==   0.002648464012203533)
    assert(isinstance(costs, float) )
    
def test_nunc_global_update_change6():
    
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
    
    np.random.seed(6)
    
    w = 20
    k = 1
    X = np.concatenate([np.repeat(10, 10), np.repeat(1, 11)])
    tree, window = NUNC.create_nunc_state_global(X[:w])
    Q = quantiles(tree, k, w)
    z_vals = [eCDF_vals(tree, q) for q in Q]
    
    current_ecdf_vals = z_vals.copy()
    
    window,z_vals,window_ecdf_vals,costs = NUNC.nunc_global_update((window, k, Q,
                                            z_vals,current_ecdf_vals),X[(w)])
     
    assert(window[w-1] == X[w]) #should have pushed out the last point
    assert(len(window) == w)
    assert(len(Q) == k)
    assert(Q[0] == 1.0)
    assert(len(z_vals) == k)
    assert(z_vals[0] == 0.2375)
    assert(costs ==   0.03601755675032692)
    assert(isinstance(costs, float) )
    
pytest.main()
