import numpy


def convert_array_8(v):
    """
        Binary flip an array of 8 elements
    """
    result = numpy.empty(8, int)
    result[0b000] = v[0b000]
    result[0b001] = v[0b100]
    result[0b010] = v[0b010]
    result[0b011] = v[0b110]
    result[0b100] = v[0b001]
    result[0b101] = v[0b101]
    result[0b110] = v[0b011]
    result[0b111] = v[0b111]
    return result
