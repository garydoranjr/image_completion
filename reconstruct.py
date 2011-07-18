#!/usr/bin/env python
from cvxopt import sparse, spmatrix, matrix as cvxmat
from numpy import nonzero, vstack, reshape, asarray, matrix, zeros, ones, ravel
from numpy.random import random, seed
from PIL import Image

from nucnrm import nrmapp

from pylab import *
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
    image_matrix = asarray(image)
    missing = missing_matrix(image_matrix, missingness)

    reconstructed = reconstruct(image_matrix, missing)
    recons_image = Image.fromarray(reconstructed, mode='L')

    dpi = rcParams['figure.dpi']
    figsize = recons_image.size[0]/dpi, recons_image.size[1]/dpi

    figure(figsize=figsize)
    ax = axes([0,0,1,1], frameon=False)
    ax.set_axis_off()
    im = imshow(recons_image, origin='lower', cmap=cm.gray, interpolation='nearest')

    show()

def missing_matrix(data, rate):
    """
    Returns a boolen matrix, where an
    entry is True if the corresponding
    value in data should be missing
    """
    return (random(data.shape) <= rate)

def reconstruct(data_matrix, missing_matrix):
    flipped = (data_matrix.shape[0] < data_matrix.shape[1])
    if flipped:
        data_matrix = data_matrix.T
        missing_matrix = missing_matrix.T

    unknown_indices = nonzero(ravel(missing_matrix, order='F'))[0]
    k = len(unknown_indices)
    A = spmatrix(1.0, unknown_indices, range(k), (data_matrix.size, k))
    known = matrix(data_matrix, dtype=float)
    known[nonzero(missing_matrix)] = 0.0
    B = cvxmat(known)
    I = spmatrix(1.0, range(k), range(k))
    G = sparse([-I, I])
    h = cvxmat(ones((2*k, 1)))
    G = -I
    h = cvxmat(ones((k, 1)))

    result = nrmapp(A, B)
    x = result['x']
    if x is not None:
        Ax = reshape(array(A*cvxmat(x)), data_matrix.shape, order='F')
        reconstructed = Ax + B
        retval = array(reconstructed, dtype=data_matrix.dtype)
        if flipped:
            return retval.T
        else:
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
