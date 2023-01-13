# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:43:57 2022

@author: austine
"""
import NUNC_module_v2 as NUNC
import pytest
import numpy as np

def test_nunc_offline_nochange1():
    
    np.random.seed(1)
    
    n = 30
    w = 20
    k = 1
    threshold = 10
    X = np.random.normal(size = n)
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "local")
    
    assert(NUNC_run.detection_time == -1)
    assert(NUNC_run.changepoint == -1)
    assert(NUNC_run.cost_max == 4.04231936038106)
    assert(len(NUNC_run.cost_history) == n - w + 1)
    assert(NUNC_run.cost_history[0] == 2.2820159898644965)
    assert(NUNC_run.data[0] == X[0])
    
def test_nunc_offline_nochange2():
    
    np.random.seed(1)
    
    n = 50
    w = 15
    k = 2
    threshold = 10
    X = np.random.normal(size = n)
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "Local")
    
    assert(NUNC_run.detection_time == -1)
    assert(NUNC_run.changepoint == -1)
    assert(NUNC_run.cost_max == 4.311099431403993)
    assert(len(NUNC_run.cost_history) == n - w + 1)
    assert(NUNC_run.cost_history[0] == 2.364529999139358)
    assert(NUNC_run.data[0] == X[0])
    
def test_nunc_offline_nochange3():
    
    np.random.seed(1)
    
    n = 100
    w = 20
    k = 2
    threshold = 10
    X = np.random.normal(size = n)
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "global")
    
    assert(NUNC_run.detection_time == -1)
    assert(NUNC_run.changepoint == -1)
    assert(NUNC_run.cost_max == 4.430417991820065)
    assert(len(NUNC_run.cost_history) == n - 2*w + 1)
    assert(NUNC_run.cost_history[0] == 1.5334303434685985)
    assert(NUNC_run.data[0] == X[0])
    
def test_nunc_offline_nochange4():
    
    np.random.seed(1)
    
    n = 100
    w = 20
    k = 2
    threshold = 100
    X = np.random.binomial(20, 0.8, size = n)
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "GLOBAL")
    
    assert(NUNC_run.detection_time == -1)
    assert(NUNC_run.changepoint == -1)
    assert(NUNC_run.cost_max == 7.81077058101598)
    assert(len(NUNC_run.cost_history) == n - 2*w + 1)
    assert(NUNC_run.cost_history[0] == 5.409660204358424)
    assert(NUNC_run.data[0] == X[0])
    
def test_NUNC_offline_change1():
    np.random.seed(1)
    
    w = 20
    k = 1
    threshold = 3
    X = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5,5,20)])
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "local")
    
    assert(NUNC_run.detection_time == 29)
    assert(NUNC_run.changepoint == 11)
    assert(NUNC_run.cost_max == 3.466955215477498)
    assert(len(NUNC_run.cost_history) == NUNC_run.detection_time - w + 1)
    assert(NUNC_run.cost_history[0] == 1.5323987762107567)
    assert(NUNC_run.data[0] == X[0])
    
def test_NUNC_offline_change2():
    np.random.seed(2)
    
    w = 10
    k = 3
    threshold = 5
    X = np.concatenate([np.random.normal(0, 1, 30), np.random.normal(-5,8,60)])
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "Local")
    
    assert(NUNC_run.detection_time == 12)
    assert(NUNC_run.changepoint == 4)
    assert(NUNC_run.cost_max == 5.544639018919987)
    assert(len(NUNC_run.cost_history) == NUNC_run.detection_time - w + 1)
    assert(NUNC_run.cost_history[0] == 3.5421991336010654)
    assert(NUNC_run.data[0] == X[0])
    
def test_NUNC_offline_change3():
    np.random.seed(2)
    
    w = 25
    k = 3
    threshold = 7
    X = np.concatenate([np.random.normal(0, 1, 30), np.random.binomial(15,0.8,60)])
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "global")
    
    assert(NUNC_run.detection_time == 52)
    assert(NUNC_run.changepoint == 27)
    assert(NUNC_run.cost_max == 7.130620129535814)
    assert(len(NUNC_run.cost_history) == NUNC_run.detection_time - 2*w + 1)
    assert(NUNC_run.cost_history[0] == 5.437512198633041)
    assert(NUNC_run.data[0] == X[0])
    
def test_NUNC_offline_change4():
    np.random.seed(2)
    
    w = 45
    k = 3
    threshold = 7
    X = np.concatenate([np.random.poisson( 1, 30), np.random.binomial(15,0.8,60)])
    
    NUNC_run = NUNC.nunc_offline(X, k, w, threshold, "GLOBAL")
    
    assert(NUNC_run.detection_time == 90)
    assert(NUNC_run.changepoint == 45)
    assert(NUNC_run.cost_max == 31.99675337798399)
    assert(len(NUNC_run.cost_history) == NUNC_run.detection_time - 2*w + 1)
    assert(NUNC_run.cost_history[0] == 31.99675337798399)
    assert(NUNC_run.data[0] == X[0])
    
def test_NUNC_offline_method_TypeError():
    
    np.random.seed(1)
    
    n = 30
    w = 20
    k = 1
    threshold = 10
    X = np.random.normal(size = n)
    
    with pytest.raises(TypeError):
        NUNC.nunc_offline(X, k, w, threshold, 1)
    
def test_NUNC_offline_method_ValueError():
    
    np.random.seed(1)
    
    n = 30
    w = 20
    k = 1
    threshold = 10
    X = np.random.normal(size = n)
    
    with pytest.raises(ValueError):
        NUNC.nunc_offline(X, k, w, threshold, "ValueError")
        

#pytest.main()
