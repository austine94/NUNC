# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:03:55 2022

@author: austine
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:02:34 2022

@author: austine
"""
import NUNC_DG as NUNC
import pytest
import numpy as np

def test_nunc_global_sequential_nochange1():
    
    np.random.seed(1)
    
    n = 50
    w = 20
    k = 1
    threshold = 10
    
    X = np.random.normal(0, 1, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)

def test_nunc_global_sequential_nochange2():
    
    np.random.seed(2)
    
    n = 30
    w = 14
    k = 3
    threshold = 10
    
    X = np.random.normal(10, 10, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)
    
def test_nunc_global_sequential_nochange3():
    
    np.random.seed(3)
    
    n = 5
    w = 2
    k = 1
    threshold = 10
    
    X = np.random.normal(-4, 2, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)
    
def test_nunc_global_sequential_nochange4():
    
    np.random.seed(4)
    
    n = 50
    w = 16
    k = 5
    threshold = 10
    
    X = np.random.poisson(10, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)
    
def test_nunc_global_sequential_nochange5():
    
    np.random.seed(5)
    
    n = 80
    w = 25
    k = 20
    threshold = 1000
    
    X = np.random.poisson(1, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)
    
def test_nunc_global_sequential_nochange6():
    
    np.random.seed(6)
    
    n = 20
    w = 5
    k = 1
    threshold = 100
    
    X = np.repeat(1, n)
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run is None)
    
def test_nunc_global_sequential_change1():
    
    np.random.seed(1)
    
    w = 10
    k = 5
    threshold = 10
    
    X = np.concatenate([np.random.normal(0, 1, 15), np.random.normal(5,5,35)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run == (20, 30, 10.159387131728794))

def test_nunc_global_sequential_change2():
    
    np.random.seed(2)
    
    w = 30
    k = 3
    threshold = 10
    
    X = np.concatenate([np.random.normal(-5, 2, 40), np.random.normal(5,5,40)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run == (32, 62, 16.266328835959396))
    
def test_nunc_global_sequential_change3():
    
    np.random.seed(3)
    
    w = 3
    k = 1
    threshold = 1
    
    X = np.concatenate([np.random.normal(0, 1, 13), np.random.normal(4,20,8)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run == (11, 14, 1.7241053534770603))
    
def test_nunc_global_sequential_change4():
    
    np.random.seed(4)
    
    w = 20
    k = 10
    threshold = 10
    
    X = np.concatenate([np.random.poisson(1, 40), np.random.poisson(4, 50)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run == (38, 58, 10.442716277037682))
    
def test_nunc_global_sequential_change5():
    
    np.random.seed(5)
    
    w = 20
    k = 19
    threshold = 10
    
    X = np.concatenate([np.random.poisson(1, 10), np.random.poisson(11, 50)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run == (20, 40, 46.09590500499422))
    
def test_nunc_global_sequential_change6():
    
    np.random.seed(6)
    
    w = 30
    k = 10
    threshold = 10
    
    X = np.concatenate([np.repeat(2, 20), np.repeat(10, 60)])
    
    NUNC_run = NUNC.nunc_global_sequential(X, k, w, threshold)
    assert(NUNC_run ==  (30, 60, 73.50832969950496))
    
#####
#Input Error Tests
#####

def test_nunc_global_sequential_k_type_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_global_sequential(X, "TypeError", 10, 10)
        
def test_nunc_globa_sequentiall_k_type_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_global_sequential(X, 9.4, 10, 10)

def test_nunc_global_sequential_k_value_error1():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, -4, 10, 10)
        
def test_nunc_global_sequential_k_value_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 0, 10, 10)
        
def test_nunc_global_sequential_k_value_error3():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 11, 10, 10)

def test_nunc_global_sequential_w_type_error1():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_global_sequential(X, 10, "TypeError", 10)
        
def test_nunc_global_sequential_w_type_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_global_sequential(X, 10, 4.4, 10)

def test_nunc_global_sequential_w_value_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 10, -6, 10)
        
def test_nunc_global_sequential_w_value_error0():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 10, 0, 10)

def test_nunc_global_sequential_threshold_type_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_global_sequential(X, 10, 10, "TypeError")
    
def test_nunc_global_sequential_threshold_value_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 10, 10, -10)
        
def test_nunc_global_sequential_threshold_value_error0():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_global_sequential(X, 10, 10, 0)
    
pytest.main()
