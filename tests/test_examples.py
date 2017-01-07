#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test all the example Jupyter notebooks.

For now we just check that they run without errors.
"""
import sys
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def _run_notebook(nb_path):
    """Helper to run a notebook and return any errors."""

    major_version = sys.version_info[0]
    assert (major_version == 2) or (major_version == 3)
    if major_version == 2:
        kernel_name = 'python'
    else:
        kernel_name = 'python3'

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(kernel_name=kernel_name)
        ep.preprocess(nb, {'metadata': {'path': 'examples/'}})

        # Check for any errors
        err = [out for cell in nb.cells if "outputs" in cell
                   for out in cell["outputs"]\
                   if out.output_type == "error"]

    return err

def test_slowness_example():
    err = _run_notebook('./examples/slowness_example.ipynb')
    assert err == []
