#!usr/bin/env python

import numpy as np
from chainer import cuda,Variable
from chainer import functions as F
import sys
import time

x_size = int(sys.argv[1]) #(x,x) size square matrix
gpu = int(sys.argv[2]) #gpumode: cpu:-1, gpu:0(gpu number)

xp = cuda.cupy if gpu>=0 else np
if gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()

def diag_part_elementwise(x): #use mask and elementwise calculation
    X = Variable(x)
    mask = Variable(xp.identity(x.shape[0],dtype=np.float32))
    return F.sum(X*mask,0)

def diag_part_batch_matmul(x): # use mask and batch_matmul calculation
    X = Variable(x)
    mask = Variable(xp.identity(x.shape[0],dtype=np.float32))
    return F.reshape(F.batch_matmul(X,mask,transa=True),(x.shape[0],))

x = xp.asarray(np.random.rand(x_size,x_size),dtype=np.float32)
cur = time.time()
A = diag_part_elementwise(x)
print("time of diag_part_elementwise:",time.time()-cur)
cur = time.time()
B = diag_part_batch_matmul(x)
print("time of diag_part_batch_matmul:",time.time()-cur)

"""
result

100x100 matrix
#cpu
$ python chainer_diag.py 100 -1
time of diag_part_elementwise: 0.0002493858337402344
time of diag_part_batch_matmul: 0.00036978721618652344
#gpu
$ python chainer_diag.py 100 0
time of diag_part_elementwise: 0.10309004783630371
time of diag_part_batch_matmul: 0.06094169616699219

1000x1000 matrix
#cpu
$ python chainer_diag.py 1000 -1
time of diag_part_elementwise: 0.0017409324645996094
time of diag_part_batch_matmul: 0.0019404888153076172
#gpu
$ python chainer_diag.py 1000 0
time of diag_part_elementwise: 0.10210251808166504
time of diag_part_batch_matmul: 0.06030750274658203

10000x10000 matrix
#cpu
$ python chainer_diag.py 10000 -1
time of diag_part_elementwise: 0.11125564575195312
time of diag_part_batch_matmul: 0.0674281120300293
#gpu
$ python chainer_diag.py 10000 0
time of diag_part_elementwise: 0.12932491302490234
time of diag_part_batch_matmul: 0.1076195240020752
"""
