""" Module for checking that types passed to functions agree with the types
indicated in the module docstring.

The docstring may specify types using the followig format

< NAME::TYPE, NAME::TYPE, NAME::TYPE... >

NAME is the variable name specified in the call signature.

TYPE may be a Python or Numpy type, or any of the following:
    TYPE[]
        indicates an iterable with elements all of TYPE
    _
        indicates any type
    TYPE ? TYPE ...
        indicates any of a set of types

TODO:
    - add method support
    - add support for Python 3 type annotations
"""

import numpy

typedict = {'int': int,
            'float': float,
            'str': str,
            'list': list,
            'tuple': tuple,
            'set': set,
            'dict': dict}

def findtypesignature(doc):
    lines = doc.splitlines()
    for line in lines[::-1]:
        if ('<' in line) and ('>' in line):
            return line
    return False

def parsetypes(doc):
    typestr = findtypesignature(doc)
    types = []
    if typestr:
        typedefs = typestr.split('<', 1)[1].rsplit('>', 1)[0]
        typedefs = [s.strip() for s in typedefs.split(',')]

        def gettype(t):
            return typedict.get(t, None)

        for td in typedefs:
            name, t = td.split('::')
            if t.endswith('[]'):
                types.append(numpy.ndarray)
            elif '?' in t:
                ts = t.split('?')
                types.append(tuple(gettype(t_) for t_ in ts))
            else:
                types.append(gettype(t))
    else:
        raise LookupError("type definitions not found within module docstring")
    return types

def verifytypes(func, bypass=False):
    """ Finds a typestring in the __doc__ of *func*, and inserts code to
    raise a TypeError is the passed arguments are not of the expected type.

    Intended to be used as a function decorator. The *bypass* keyword argument
    can be used to disable typechecking for code believed to be safe.
    """
    exptypes = parsetypes(func.__doc__)
    if bypass:
        return func

    def wrapper(*args):

        def checktype(et, a):
            if not isinstance(et, type):
                if type(a) not in et:
                    raise TypeError("expected one of {0} but "
                                    "received {1}".format(et, type(a)))
            elif type(a) != et:
                raise TypeError("expected {0} but "
                                "received {1}".format(et, type(a)))
            return 1

        for arg, exptype in zip(args, exptypes):
                checktype(exptype, arg)

        func(*args)

    return wrapper


