"""Methods to check types of variables."""

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
