from random import random
import time
import numpy as np
import numba
from numba import prange
from numba import jit
from numba import cuda

@jit (nopython=True, cache=True)
def monte_carlo_pi_cpu(itr=10000):
    hit_count = 0
    for _ in range(itr):
        x = random()
        y = random()
        if (x ** 2 + y ** 2) < 1.0:
            hit_count += 1
    result = 4.0 * hit_count / itr
    return result

@jit (nopython=True, parallel=True, cache=True)
def monte_carlo_pi_parallel(itr=10000):
    hit_count = 0
    for _ in range(itr):
        x = random()
        y = random()
        if (x ** 2 + y ** 2) < 1.0:
            hit_count += 1
    result = 4.0 * hit_count / itr
    return result

@jit (nopython=True, parallel=True, cache=True)
def vector_add_parallel(x, y, size):
    for i in prange(size):
        y[i] = x[i] + y[i]

@cuda.jit
def vector_add_cuda(x, y, size):
    i = cuda.grid(1)

    i_in_bounds = (i >= 0) and (i < size)

    if i_in_bounds:
        y[i] = x[i] + y[i]

print(numba.__version__)
cuda.detect()

size = 1024 * 1024
a = np.ones(size)
b = np.ones(size)

start_time = time.time()
vector_add_parallel(a, b, size)
duration = time.time() - start_time
print('{}\n\nTake Times: {}\n'.format(b, duration))

a = np.ones(size)
b = np.ones(size)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)

block_size = 128
grid_size = size // 128

start_time = time.time()
vector_add_cuda[grid_size, block_size](d_a, d_b, size)
duration = time.time() - start_time
output = d_b.copy_to_host()
print('{}\n\nTake Times: {}\n'.format(output, duration))