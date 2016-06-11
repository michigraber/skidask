
#   import multiprocessing
#   try:
#       multiprocessing.set_start_method('forkserver')
#   except:
#       pass

import h5py
import numpy as np

from skimage.util import apply_parallel
from skimage import filters

from memory_profiler import profile

import dask.array as da


@profile
def run_apply_parallel(imagepath, chunks, mode='h5py-direct'):

    sigma = 3

    if mode == 'h5py-direct':
        print('Running gaussing filter on h5py array using dask.')
        with h5py.File(imagepath, 'r') as f:
            image = f['data']
            result = apply_parallel(filters.gaussian, image, chunks, extra_arguments=(sigma, ), depth=sigma)

    elif mode == 'numpy-array':
        print('Running gaussing filter on numpy array using dask.')
        with h5py.File(imagepath, 'r') as f:
            image = f['data'][:]
            result = apply_parallel(filters.gaussian, image, chunks, extra_arguments=(sigma, ), depth=sigma)

    return result



@profile
def run_map_overlap(func, imagepath, chunks, overlap, mode='h5py-direct', extra_arguments=(), extra_keywords={}):

    boundary='periodic'

    def wrapped_func(arr):
        return func(arr, *extra_arguments, **extra_keywords)

    if mode == 'h5py-direct':
        print('Running gaussing filter on h5py array using dask.')
        with h5py.File(imagepath, 'r') as f:
            darr = da.from_array(f['data'], chunks=chunks)
            result = darr.map_overlap(wrapped_func, overlap, boundary=boundary).compute()

    elif mode == 'numpy-array':
        print('Running gaussing filter on numpy array using dask.')
        with h5py.File(imagepath, 'r') as f:
            darr = da.from_array(f['data'][:], chunks=chunks)
            result = darr.map_overlap(wrapped_func, overlap, boundary=boundary).compute()

    return result
