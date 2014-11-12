import integerfft
import numpy
import random

rmatrix = numpy.matrix([[random.randint(0, 255) for x in xrange(8)] for x
                        in xrange(8)])


def dct_8x8(m):
    tm = numpy.matrix([integerfft.dct_8(m[k].A1) for k in xrange(m.shape[0])])
    return numpy.hstack([numpy.matrix(integerfft.dct_8(tm[:, k].A1)).T
                         for k in xrange(tm.shape[1])])


def inverse_dct_8x8(m):
    tm = numpy.hstack([numpy.matrix(integerfft.inverse_dct_8(m[:, k].A1)).T
                       for k in xrange(m.shape[1])])
    return numpy.matrix([integerfft.inverse_dct_8(tm[k].A1)
                         for k in xrange(tm.shape[0])])


def compute_8x8_patches(m, function):
    height = m.shape[0]
    width = m.shape[1]
    if (width % 8 != 0 or height % 8 != 0):
        raise "Matrix dimensions should be multiple of 8"
    result = numpy.matrix([[0]*width]*height)
    for y in xrange(0, height, 8):
        for x in xrange(0, width, 8):
            result[y:y+8, x:x+8] = function(m[y:y+8, x:x+8])
    return result


def dct(m):
    return compute_8x8_patches(m, dct_8x8)


def inverse_dct(m):
    return compute_8x8_patches(m, inverse_dct_8x8)
