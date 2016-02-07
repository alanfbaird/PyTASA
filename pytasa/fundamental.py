# -*- coding: utf-8 -*-
"""
pytasa.fundamental - basic operations with elastic constant matrices

This module provides functions that operate on single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np

# FIXME: this is a good place to put things like phase velocity 
#        calculation or tensor rotation. Basic rule should be that
#        the functions take a 6x6 np. array (or matrix?) as input
#        and return something. We can then build an OO interface on top
