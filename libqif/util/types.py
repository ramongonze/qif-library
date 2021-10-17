"""Methods to check types of variables."""

from numpy import array

def check_int(value):
    if type(value) != type(1):
        raise TypeError('The value must be an integer')
    else:
        return value

def check_string(value):
    if type(value) != type('string'):
        raise TypeError('The value must be a string')
    else:
        return value

def check_float(value):
    if type(value) != type(1.0):
        raise TypeError('The value must be a float')
    else:
        return value

def check_list(value):
    if type(value) != type([]):
        raise TypeError('The value must be a list')
    else:
        return value

def check_dict(value):
    if type(value) != type(dict()):
        raise TypeError('The value must be a dictionary')
    else:
        return value

def check_set(value):
    if type(value) != type(set()):
        raise TypeError('The value must be a set')
    else:
        return value

def check_numpy_array(value):
    if type(value) != type(array([])):
        raise TypeError('The value must be a numpy array')
    else:
        return value