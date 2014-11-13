import numpy
import PIL.Image
import sys
import huffman
import matrix

if (len(sys.argv) < 2):
    print "Usage: " + sys.argv[0] + " [Image File]"
    exit(-1)

m = numpy.asmatrix(PIL.Image.open(sys.argv[1]))
print "Estimating Huffman Size ... "
size_before = huffman.estimate(m.A1)/8 + 1
print "Estimated Huffman Size:", \
    size_before, "bytes"
print "Transforming matrix ..."
transformed_m = matrix.dct(m)
print "Done. Estimating Transformed Huffman Size ..."
size_after = huffman.estimate(transformed_m.A1)/8 + 1
print "Estimated Transformed Huffman Size:", \
    size_after, "bytes"
print "Compression ratio:", float(size_after)/float(size_before)
print "Inverting Transformation ..."
untransformed_m = matrix.inverse_dct(transformed_m)
if numpy.array_equal(untransformed_m, m):
    print "Inverted transformation equals original matrix"
else:
    print "Error. Inverted transformation does not equal original matrix"
