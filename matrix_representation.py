import stk
import torch

"""
Purpose: understand how the stk represents sparse matrix
"""

"""
op: transpose or non-transpose

[Sparse Matrix Multiplication]
stk.ops.dsd: dense = op(sparse) x op(dense)
stk.ops.dds: dense = op(dense) x op(sparse)
stk.ops.sdd: sparse = op(dense) x op(dense)

[Sparse Matrix Conversion]
stk.ops.to_sparse: torch.Tensor => stk.Matrix
stk.ops.to_dense: stk.Matrix => torch.Tensor

[Sparse Matrix Generation]
stk.random.dense_mask: Create a random, block-sparse dense matrix.
stk.random.mask: Create a random, block-sparse sparse matrix.
"""

A=torch.rand(10,10)
#def dense_mask(rows, cols, sparsity, blocking=1):
B=stk.random.mask(10,10,.5)

print("Overall Matrix Dimension")
print(B.shape)

print("Dense Matrix")
print(stk.ops.to_dense(B).numpy())

print("Sparse Matrix Data")
print(B.data.numpy().squeeze())

print("Offsets -> the i where the row starts")
print(B.offsets.numpy().shape)
print(B.offsets.numpy())

print("Row Indices -> row_indices[i] ~ the row data[i] belong to")
print(B.row_indices.numpy().shape)
print(B.row_indices.numpy())

print("Column Indices -> row_indices[i] ~ the column data[i] belong to")
print(B.column_indices.numpy().shape)
print(B.column_indices.numpy())

print("Offsets_t")
print(B.offsets_t.numpy().shape)
print(B.offsets_t.numpy())

print("Column_indices_t")
print(B.column_indices_t.numpy().shape)
print(B.column_indices_t.numpy())

print("Block_offsets_t")
print(B.block_offsets_t.numpy().shape)
print(B.block_offsets_t.numpy())
