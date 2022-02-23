# -*- coding: utf-8 -*-
import copy
import itertools


def transform_to_sequence_of_arguments(npoints, *args):
    """

    Parameters
    ----------
    npoints : int
    *args : float | (float, float) | dict[Any, float | (float, float)]

    Returns
    -------
    List[Iterable[float] | Iterable[dict[Any, float]]]

    """
    res = []
    for arg in args:
        if _is_number(arg):
            res.append(itertools.repeat(arg, npoints))
        elif _is_sequence(arg) and not isinstance(arg, dict):
            res.append(_linspace_iterator(arg[0], arg[1], npoints))
        elif isinstance(arg, dict):
            res.append(_dict_iterator(arg, npoints))
    return res


def _is_sequence(obj):
    try:
        len(obj)
    except:
        return False
    else:
        return True
    
    
def _is_number(obj):
    try:
        float(obj)
    except:
        False
    else:
        return True
    
    
def _linspace_iterator(a, b, n):
    for i in range(n):
        yield a + (b - a)/(n-1)*i
        

def _dict_iterator(d, npoints):
    d_copy = copy.copy(d)
    for key, val in d_copy.items():
        val_iterator = itertools.repeat(val, npoints) if _is_number(val) \
                         else _linspace_iterator(val[0], val[1], npoints)
        d_copy[key] = val_iterator
    for i in range(npoints):
        yield {k: next(v) for k, v in d_copy.items()}
    