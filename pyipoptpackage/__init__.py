"""
This is a package for a python interface to ipopt.

The underlying C interface is in pyipoptcore.
"""

import functools

import numpy

from pyipopt.ipoptconst import *
from pyipopt.pyipoptcore import *
from pyipopt.ipoptunconstrained import fmin_unconstrained

# verbose messages from the C interface
set_loglevel(2)

