#!/usr/bin/env python
from cvxopt import spmatrix, matrix as cvxmat
import numpy as np
from numpy import nonzero, reshape, asarray, matrix, ravel, array
from numpy.random import random, seed
from PIL import Image

from matplotlib.pyplot import figure, cm, show

from nucnrm import nrmapp

def main(filename, missingness, width, height):
    """
    Usage:

        ./reconstruct.py filename missingness [width [height]]

    Adds random missingness to image specified by filename, then uses
    nuclear norm minimization to reconstruct the image. If width and
    height are specified, the image is first rescaled. If only width
    is specified, the image is scaled proportionally.

    The missingness argument should be a number in [0, 1), indicating
    the percentage of missing pixels. The width and height arguments
    should be integers.
    """
    seed(12345)
    image = Image.open(filename)
    image = image.convert('L')
    if width is not None:
        if height is None:
            height = width*image.size[1]/image.size[0]
        image = image.resize((width, height))
    image_matrix = array(asarray(image))
    missing = missing_matrix(image_matrix, missingness)

    reconstructed = reconstruct(image_matrix, missing)

    fig = figure()
    sp_args = { 'frameon': False,
                'xticks': [],
                'yticks': [] }
    im_args = { 'cmap': 'gray',
                'vmin': 0,
                'vmax': 255,
                'interpolation': 'nearest' }
    # Original
    ax = fig.add_subplot(221, **sp_args)
    im = ax.imshow(image_matrix, **im_args)

    # Reconstructed
    ax = fig.add_subplot(222, **sp_args)
    im = ax.imshow(reconstructed, **im_args)

    # Missing
    missing_data = array(asarray(image.convert('RGB')))
    for x, y in zip(*nonzero(missing)):
        missing_data[x, y, :] = (0xFF, 0, 0)
    ax = fig.add_subplot(223, **sp_args)
    im = ax.imshow(missing_data, interpolation='nearest')

    # Difference
    diff = np.abs(array(reconstructed, dtype=float) -
                  array(image_matrix, dtype=float))
    diff = array(diff, dtype=image_matrix.dtype)
    ax = fig.add_subplot(224, **sp_args)
    im = ax.imshow(diff, **im_args)

    show()

def missing_matrix(data, rate):
    """
    Returns a boolen matrix, where an entry is True if the
    corresponding value in data should be missing.
    """
    return (random(data.shape) <= rate)

def reconstruct(data_matrix, missing_matrix):
    """
    Reconstructs the data_matrix after the pixels indicated in the
    missing_matrix have been removed.
    """
    flipped = (data_matrix.shape[0] < data_matrix.shape[1])
    if flipped:
        data_matrix = data_matrix.T
        missing_matrix = missing_matrix.T

    unknown_indices = nonzero(ravel(missing_matrix, order='F'))[0]
    known = matrix(data_matrix, dtype=float)
    known[nonzero(missing_matrix)] = 0.0
    u = len(unknown_indices)
    A = spmatrix(1.0, unknown_indices, range(u), (data_matrix.size, u))
    B = cvxmat(known)

    x = nrmapp(A, B)['x']
    if x is not None:
        Ax = reshape(array(A*cvxmat(x)), data_matrix.shape, order='F')
        retval = array(Ax + B, dtype=data_matrix.dtype)
        if flipped:
            retval = retval.T
        return retval
    else:
        raise Exception("Error during optimization.")

def print_usage(quit=True):
    print main.__doc__
    if quit: exit()

def safe_pop(arg_list, mapper=lambda x: x):
    try:
        return mapper(arg_list.pop(0))
    except IndexError:
        return None

if __name__ == '__main__':

    # Parse arguments
    from sys import argv
    args = argv[1:]
    if len(args) < 2 or len(args) > 4:
        print_usage()
    else:
        try:
            filename = safe_pop(args)
            missingness = safe_pop(args, float)
            width = safe_pop(args, int)
            height = safe_pop(args, int)
        except:
            print_usage()

    if missingness >= 1.0 or missingness < 0:
        print_usage()

    main(filename, missingness, width, height)
