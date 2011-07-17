#!/usr/bin/env python

def main(filename, missingness, width, height):
    """
    Usage:

        ./reconstruct.py filename missingness [width [height]]

    Adds random missingness to image specified by filename, then uses
    nuclear norm minimization to reconstruct the image. If width and
    height are specified, the image is first rescaled. If only width
    is specified, the image is scaled proportionally.

    The missingness argument should be a number in [0, 1], indicating
    the percentage of missing pixels. The width and height arguments
    should be integers.
    """
    pass

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

    main(filename, missingness, width, height)
