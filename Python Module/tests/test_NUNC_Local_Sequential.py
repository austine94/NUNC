# -*- coding: utf-8 -*-
"""
Created on Mon May 23 21:29:12 2022

@author: austine
"""
import NUNC_module_v2 as NUNC
import pytest
import numpy as np

def test_nunc_local_nochange1():
    
    np.random.seed(1)
    
    n = 30
    w = 20
    k = 1
    threshold = 10
    X = np.random.normal(size = n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)

def test_nunc_local_nochange2():
    
    np.random.seed(2)
    
    n = 42
    w = 40
    k = 7
    threshold = 35
    X = np.random.normal(size = n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)
    
def test_nunc_local_nochange3():
    
    np.random.seed(3)
    
    n = 3
    w = 2
    k = 2
    threshold = 30
    X = np.random.normal(100, 40, n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)
    
def test_nunc_local_nochange4():
    
    np.random.seed(4)
    
    n = 50
    w = 50
    k = 10
    threshold = 49
    X = np.random.poisson(2, n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)

def test_nunc_local_nochange5():
    
    np.random.seed(5)
    
    n = 30
    w = 10
    k = 3
    threshold = 100
    X = np.repeat(10, n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)
    
def test_nunc_local_nochange6():
    
    np.random.seed(6)
    
    n = 30
    w = 29
    k = 28
    threshold = 100
    X = np.random.binomial(10, 0.6, n)
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    
    assert(NUNC_run is None)
    
def test_nunc_local_change1():
    
    np.random.seed(1)
    
    w = 20
    k = 1
    threshold = 3
    X = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5,5,20)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (11, 29, 3.466955215477498))

def test_nunc_local_change2():
    
    np.random.seed(2)
    
    w = 35
    k = 4
    threshold = 3
    X = np.concatenate([np.random.normal(0, 1, 15), np.random.normal(15,10,20)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (31, 35, 6.002951403704943))

def test_nunc_local_change3():
    
    np.random.seed(3)
    
    w = 5
    k = 2
    threshold = 2
    X = np.concatenate([np.random.binomial(7, 0.4, 15), np.random.normal(2,10,40)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (3, 6, 2.0339365992551985))


def test_nunc_local_change4():
    
    np.random.seed(4)
    
    w = 30
    k = 10
    threshold = 10
    X = np.concatenate([np.random.binomial(8, 0.9, 15), np.random.binomial(2,0.2,40)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (13, 30, 25.875036671558476))
    
def test_nunc_local_change5():
    
    np.random.seed(5)
    
    w = 100
    k = 10
    threshold = 10
    X = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10,5,100)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (127, 135, 10.12956067169531))
    
def test_nunc_local_change6():
    
    np.random.seed(6)
    
    w = 20
    k = 1
    threshold = 4
    X = np.concatenate([np.repeat(30, 10), np.repeat(0,20)])
    
    NUNC_run = NUNC.nunc_local(X, k, w, threshold)
    assert(NUNC_run == (9, 20, 5.232481437645479))
    
#####
#Input Error Tests
#####

def test_nunc_local_k_type_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_local(X, "TypeError", 10, 10)
        
def test_nunc_local_k_type_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_local(X, 9.4, 10, 10)

def test_nunc_k_value_error1():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc(X, -4, 10, 10)
        
def test_nunc_local_k_value_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 0, 10, 10)
        
def test_nunc_local_k_value_error3():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 11, 10, 10)

def test_nunc_local_w_type_error1():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_local(X, 10, "TypeError", 10)
        
def test_nunc_local_w_type_error2():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc(X, 10, 4.4, 10)

def test_nunc_local_w_value_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 10, -6, 10)
        
def test_nunc_local_w_value_error0():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 10, 0, 10)

def test_nunc_local_threshold_type_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.nunc_local(X, 10, 10, "TypeError")
    
def test_nunc_local_threshold_value_error():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 10, 10, -10)
        
def test_nunc_local_threshold_value_error0():
    
    X = np.random.normal(size = 20)

    with pytest.raises(ValueError):
        NUNC.nunc_local(X, 10, 10, 0)
    
pytest.main()
