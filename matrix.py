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
