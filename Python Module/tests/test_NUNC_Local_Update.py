# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:21:58 2022

@author: austine
"""
#NUNC Local Update Unit Tests
#So far, I have changed the start of the NUNC_Local function to X[w-1]
#and I have changed max_length of the deque to be one greater as you create
#the state when you haave a window of length w-1, and so there is room for
#one more point.

import NUNC_module_v2 as NUNC
import pytest
import numpy as np

def test_nunc_local_update_nochange1():
    
    np.random.seed(1)
    
    w = 20
    k = 1
    X = np.random.normal(size = w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[10] == np.sort(X)[10])
    assert(window[19] == X[19])
    assert(len(window) == w)
    assert(max(costs) ==  2.2820159898644965)
    assert(len(costs) == w-1) 
    
def test_nunc_local_update_nochange2():    
    
    np.random.seed(2)
    
    w = 40
    k = 3
    X = np.random.normal(size = w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[14] == np.sort(X)[14])
    assert(window[39] == X[39])
    assert(len(window) == w)
    assert(max(costs) ==  4.282898114802451) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_nochange3():    
    
    np.random.seed(3)
    
    w = 3
    k = 2
    X = np.random.normal(size = w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[0] == np.sort(X)[0])
    assert(window[2] == X[2])
    assert(len(window) == w)
    assert(costs[0] ==  1.150728289807124) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_nochange4():    
    
    np.random.seed(4)
    
    w = 50
    k = 40
    X = np.random.normal(size = w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[0] == X[0])
    assert(len(window) == w)
    assert(costs[w-2] ==  5.7954395816064554) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_nochange5():    
    
    np.random.seed(5)
    
    w = 20
    k = 2
    X = np.random.binomial(10, 0.6, size = w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[0] == X[0])
    assert(len(window) == w)
    assert(min(costs) == 0.004742360773768439) 
    assert(len(costs) == w-1)
 
def test_nunc_local_update_nochange6():    
    
    np.random.seed(6)
    
    w = 20
    k = 2
    X = np.repeat(10, w)
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[0] == X[0])
    assert(len(window) == w)
    assert(max(costs) == 0.0) 
    assert(len(costs) == w-1)

def test_nunc_local_update_change1():    
    
    np.random.seed(1)
    
    w = 20
    k = 2
    X = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5,5,10)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[4] == np.sort(X)[4])
    assert(window[3] == X[3])
    assert(len(window) == w)
    assert(max(costs) == 3.7043437983805667) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_change2():    
    
    np.random.seed(2)
    
    w = 16
    k = 7
    X = np.concatenate([np.random.normal(0, 1, 8), np.random.normal(-3,12,8)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[2] == X[2])
    assert(len(window) == w)
    assert(min(costs) == -1.1605114981038858) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_change3():    
    
    np.random.seed(3)
    
    w = 41
    k = 1
    X = np.concatenate([np.random.normal(-2,2,25), np.random.normal(5,2,16)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[2] == X[2])
    assert(len(window) == w)
    assert(costs[0] == 0.04876524072563271) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_change4():    
    
    np.random.seed(4)
    
    w = 30
    k = 2
    X = np.concatenate([np.repeat(0, 15), np.repeat(20, 15)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[0] == np.sort(X)[0])
    assert(window[12] == X[12])
    assert(len(window) == w)
    assert(costs[28] == 0.8605489869558518) 
    assert(len(costs) == w-1)
    
def test_nunc_local_update_change5():    
    
    np.random.seed(5)
    
    w = 19
    k = 3
    X = np.concatenate([np.repeat(0, 7), np.repeat(20, 12)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[10] == np.sort(X)[10])
    assert(window[0] == X[0])
    assert(len(window) == w)
    assert(max(costs) == 7.682470458362806) 
    assert(len(costs) == w-1)

def test_nunc_local_update_change6():    
    
    np.random.seed(6)
    
    w = 20
    k = 5
    X = np.concatenate([np.random.binomial(12,0.4,10), np.random.binomial(2,0.1,10)])
    tree,window = NUNC.create_nunc_state(X[:(w-1)])
    
    
    tree,window,costs = NUNC.nunc_local_update((tree, window, k), X[(w-1):])
    
    assert(tree[w-1] == np.sort(X)[w-1])
    assert(window[2] == X[2])
    assert(len(window) == w)
    assert(max(costs) == 8.730761942017672) 
    assert(len(costs) == w-1)
    
    
#pytest.main()
