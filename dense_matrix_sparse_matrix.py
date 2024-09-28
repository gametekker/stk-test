import stk
import torch
from stk.ops.linear_ops_test import _generate_testcases
import torch.autograd.profiler as profiler
from copy import deepcopy
import time

"""
Purpose: demonstrate how to use stk.ops.dds to perform sparse matrix multiplication
#takeaway 1: it does NOT appear faster than torch.mm
#takeaway 2: stk.ops.dds is built with the graph in mind, so it can be used to calculate gradients
"""

params = _generate_testcases()[10]
params = (4096, 4096, 4096, 0.90, False, True, 128, torch.float16)
print(params)
m, n, p, sparsity, trans_a, trans_b, blocking, dtype = params

cuda_device = torch.device("cuda")
std=1.0

# Construct the operands.

#1. use torch to generate a random dense matrix A
a_baseline = (torch.randn(m, n) * std).type(dtype)

#2. use torch to generate a random dense matrix B and apply a mask to make it sparse 
mask = stk.random.dense_mask(n, p, sparsity, blocking)
b_baseline = (torch.randn(n, p) * std * mask).type(dtype)

# Execute the matmul.
#NOTE: before performing matmul, must set requires_grad=True

#3. use stk.ops.dds to calculate the experiment result
#NOTE: before using stk.ops.dds, must convert b_baseline to stk.Matrix
a=deepcopy(a_baseline)
b=stk.ops.to_sparse(b_baseline, blocking)
print(b.data.shape) #for illustration purpose
a=a.to(cuda_device).requires_grad_(True)
b=b.to(cuda_device).requires_grad_(True)
begin=time.time()
c_experiment=stk.ops.dds(a, b)
end=time.time()
print(f'exp: {end-begin}')

#4. TODO: use CUDA kernel for dense sparse matrix multiply

#5. use torch torch.mm to calculate the baseline result
a_baseline=a_baseline.to(cuda_device).requires_grad_(True)
b_baseline=b_baseline.to(cuda_device).requires_grad_(True)
begin=time.time()
c_baseline = torch.mm(a_baseline, b_baseline)
end=time.time()
print(f'base: {end-begin}')

print(c_experiment==c_baseline)

# Compute the gradients w.r.t. the inputs.
#c_baseline.sum().backward()
#c_experiment.sum().backward()
