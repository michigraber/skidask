
from memory_profiler import profile

import h5py
import dask.array as da


@profile
def run_map_overlap(func, imagepath, chunks, overlap, mode='h5py-direct', extra_func_args=(), extra_func_kwargs={}):

    boundary='periodic'

    def wrapped_func(arr):
        return func(arr, *extra_func_args, **extra_func_kwargs)

    if mode == 'h5py-direct':
        print('Running func on h5py array using dask.')
        with h5py.File(imagepath, 'r') as f:
            darr = da.from_array(f['data'], chunks=chunks)
            result = darr.map_overlap(wrapped_func, overlap, boundary=boundary).compute()

    elif mode == 'numpy-array':
        print('Running func on numpy array using dask.')
        with h5py.File(imagepath, 'r') as f:
            darr = da.from_array(f['data'][:], chunks=chunks)
            result = darr.map_overlap(wrapped_func, overlap, boundary=boundary).compute()

    return result


def add_x(arr, x):
    return arr+x
