# -*- coding: utf-8 -*-
"""
Created on Sun May 15 09:52:12 2022

@author: austine
"""
import pytest
import numpy as np
import NUNC_module as NUNC

#####################
#Check NUNC_offline returns correct outputs under the null
#####################

def test_NUNC_Local_null():
    #checks output for NUNC Local under null
    
    np.random.seed(1)
    data = np.random.normal(size = 100)
    
    NUNC_run = NUNC.NUNC_offline(data, 100, 50, 10, method = "Local")
    
    assert NUNC_run.changepoint == -1
    assert NUNC_run.detection_time is None
    assert NUNC_run.cost_vec[50] == 4.6773857174039275
    assert NUNC_run.method == "local"
    assert NUNC_run.max_checks == 50
    assert NUNC_run.data[32] == data[32]

    
def test_NUNC_Local_Grid_null():
    
    np.random.seed(1)
    data = np.random.normal(size = 100)
    
    NUNC_run = NUNC.NUNC_offline(data, 100, 50, 10, method = "Local",
                                 max_checks = 5)
    
    assert NUNC_run.changepoint == -1
    assert NUNC_run.detection_time is None
    assert NUNC_run.cost_vec[50] == 4.6773857174039275
    assert NUNC_run.method == "local"
    assert NUNC_run.max_checks == 5
    assert NUNC_run.data[32] == data[32]

    
def test_NUNC_Global_null():
    #checks output for NUNC Global
    
    np.random.seed(1)
    data = np.random.normal(size = 200)
    
    NUNC_run = NUNC.NUNC_offline(data, 100, 50, 10, method = "global")
    
    assert NUNC_run.changepoint == -1
    assert NUNC_run.detection_time is None
    assert NUNC_run.cost_vec[50] == 3.6325196916624236
    assert NUNC_run.method == "global"
    assert NUNC_run.max_checks == 1
    assert NUNC_run.data[32] == data[32]

    
def test_NUNC_Semiparametric_null():
    #checks output for NUNC Global
    
    np.random.seed(1)
    data = np.random.normal(size = 200)
    
    prob_vec = np.arange(0.1, 0.99, step = 0.09)
    q_vec = np.quantile(data, prob_vec)
    
    NUNC_run = NUNC.NUNC_offline(data, 100, 50, 10, "semiparametric",
                                 quantiles = q_vec)
    
    assert NUNC_run.changepoint == -1
    assert NUNC_run.detection_time is None
    assert NUNC_run.cost_vec[50] == 0.33043461602765856
    assert NUNC_run.method == "semiparametric"
    assert NUNC_run.max_checks == 1
    assert NUNC_run.data[32] == data[32]
    
##############
#Check functions work with a changepoint in the interval
##############

def test_NUNC_Local_Change():
    
    np.random.seed(2)
    data = np.concatenate([np.random.normal(size = 75),
                          np.random.normal(loc = 5, size = 75)])
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 50, 5)
    
    assert NUNC_run.changepoint == 75
    assert NUNC_run.detection_time == 94
    
def test_NUNC_Local_Grid_Change():
    
    np.random.seed(2)
    data = np.concatenate([np.random.normal(size = 75),
                          np.random.normal(loc = 5, size = 75)])
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 50, 5, max_checks = 5)
    
    assert NUNC_run.changepoint == 75
    assert NUNC_run.detection_time == 99
    
def test_NUNC_Global_Change():
    
    np.random.seed(2)
    data = np.concatenate([np.random.normal(size = 75),
                          np.random.normal(loc = 5, size = 75)])
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 50, 5, method = "global")
    
    assert NUNC_run.changepoint == 65
    assert NUNC_run.detection_time == 115
    
def test_NUNC_Semiparametric_Change():
    
    np.random.seed(2)
    data = np.concatenate([np.random.normal(size = 75),
                          np.random.normal(loc = 5, size = 75)])
    
    prob_vec = np.arange(0.1, 0.99, step = 0.18)
    q_vec = np.quantile(data[:50], prob_vec)
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 50, 5, method = "semiparametric",
                                 quantiles = q_vec)
    
    assert NUNC_run.changepoint == 60
    assert NUNC_run.detection_time == 110
    
###################
#Test NUNCoffline input checks
###################

def test_NUNC_data_TypeError():
    #checks data only accepts ndarray
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(33, 10, 50, 5, method = "global")
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(list([10, 12, 15]), 10, 50, 5, method = "global")
    with pytest.raises(TypeError):
        NUNC.NUNC_offline("anomaly?", 10, 50, 5, method = "local")

def test_NUNC_threshold_TypeError():
    #checks threshold must be a float (or valid as float)
    data = np.random.normal(size = 20)
    
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, "anomaly", 10, 5)

def test_NUNC_threshold_ValueError():
    #checks threshold must be postive
    data = np.random.normal(size = 20)
    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, -6, 10, 5)    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 0, 10, 5) 
    
def test_NUNC_K_TypeError():
    #checks K must be an int
    data = np.random.normal(size = 20)
    
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 10, "anomaly?")
        
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 10, 5.0)
        
def test_NUNC_K_ValueError():
    #checks K is positive and less than w
    
    data = np.random.normal(size = 20)
    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 12, -6)
        
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 10, 0)
        
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 5, 8)

    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 5, 5)
    
def test_NUNC_w_ValueError():
    #checks w is positive and does not exceed data length
    data = np.random.normal(size = 20)
    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, -6, 5)
        
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 0, 5)
        
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 21, 5)
        
def test_NUNC_w_TypeError():
    #checks K must be an int
    data = np.random.normal(size = 20)
    
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, "anomaly?", 5)
        
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 14.0, 5)
        
def test_NUNC_method_TypeError():
    #checks method is a string
    data = np.random.normal(size = 20)
    
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 12, 5, 22.0)
        
def test_NUNC_method_ValueError():
    #checks method is valid and .lower() works on input string
    data = np.random.normal(size = 20)
    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 12, 5, "anomaly?")
        
    NUNC_run = NUNC.NUNC_offline(data, 10, 12, 5, "LOCAL")
    assert NUNC_run.method == "local"    
    
    
def test_NUNC_maxchecks_TypeError():
    #checks max checks is correct type
    data = np.random.normal(size = 20)
    
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 12, 5, "local", max_checks = "anomaly?")
        
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 12, 5, "local", max_checks = 14.0)
        
def test_NUNC_maxchecks_ValueError():
    #checks max_checks must be positive integer
    data = np.random.normal(size = 20)
    
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 12, 5, "local", max_checks = -4)
        
    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 12, 5, "local", max_checks = 0)
    
def test_NUNC_maxchecks_Global():
    #checks max_checks does not alter Global or semiparametric
    data = np.random.normal(size = 20)
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 8, 3, "global", max_checks = 3)
    assert NUNC_run.max_checks == 1
    
    prob_vec = np.arange(0.1, 0.99, step = 0.18)
    q_vec = np.quantile(data, prob_vec)
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 8, 5, "semiparametric",
                                 max_checks = 3, quantiles = q_vec)
    assert NUNC_run.max_checks == 1
    
def test_NUNC_maxchecks_large_grid():
    #checks max_checks is reduced to w if it is specified as greater than w
    data = np.random.normal(size = 20)
    
    NUNC_run = NUNC.NUNC_offline(data, 10, 8, 3, "local", max_checks = 9)
    assert NUNC_run.max_checks == 8
    
def test_NUNC_quantiles_TypeError():
    #checks that quantiles input must be ndarray
    data = np.random.normal(size = 20)

    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 8, 5, "semiparametric", quantiles = 14.0)
        
    with pytest.raises(TypeError):
        NUNC.NUNC_offline(data, 10, 8, 5, "semiparametric",
                          quantiles = "anomaly?")
        
def test_NUNC_quantiles_ValueError():
    #checks length of quantiles array are equal to K
    data = np.random.normal(size = 20)
    
    prob_vec = np.arange(0.1, 0.99, step = 0.09)
    q_vec = np.quantile(data, prob_vec)

    with pytest.raises(ValueError):
        NUNC.NUNC_offline(data, 10, 8, 5, "semiparametric", quantiles = q_vec)

        
#pytest.main()
