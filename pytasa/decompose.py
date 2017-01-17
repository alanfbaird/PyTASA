# -*- coding: utf-8 -*-
"""
pytasa.decompose - decompose tensors following Browaeys & Chevrot (2004)

This module provides functions that implement the decomposition of elasticity
tensors according to their symmetry using the approach described by Browaeys
and Chevrot (2004)

References:

    Browaeys, J. T. and S. Chevrot (2004) Decomposition of the elastic
             tensor and geophysical applications. Geophysical Journal 
             international v159, 667-678
"""
import collections
import numpy as np

ElasticNorms = collections.namedtuple('ElasticNorms', ['isotropic', 'trigonal',
                    'hexagonal', 'orthorhombic', 'monoclinic', 'triclinic'])
def norms(C1, C2, C3, C4, C5, C6, C7):
    result = ElasticNorms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return result
