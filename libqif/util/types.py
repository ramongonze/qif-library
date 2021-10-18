"""Methods to check types of variables."""

from numpy import array, arange

def is_int(value):
    return type(value) == type(1)

def is_float(value):
    return type(value) == type(1.0)

def is_string(value):
    return type(value) == type('string')

def is_list(value):
    return type(value) == type([])

def is_dict(value):
    return type(value) == type(dict())

def is_set(value):
    return type(value) == type(set())

def is_numpy_array(value):
    return type(value) == type(array([]))

def is_2d_list_matrix(value):
    if type(value) != type([]):
        return False
    
    for i in arange(len(value)):
        if type(value[i]) != type([]):
            return False
    
    return True

def is_2d_numpy_matrix(value):
    return type(value) == type(array([])) and value.ndim == 2

def is_function(value):
    return type(value) == type(lambda x : None)