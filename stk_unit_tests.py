"""
unit test
"""

from stk.ops.linear_ops_test import LinearOpsTest
import torch
import unittest

suite = unittest.TestSuite()
suite.addTest(LinearOpsTest('testLinearOps_Dsd0'))  # Specify the test case
runner = unittest.TextTestRunner()
runner.run(suite)