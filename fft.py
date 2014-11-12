import cmath


def dct_butterfly((x, y), (N, j)):
    c = math.cos(math.pi*j/(2.0*N))
    s = math.sin(math.pi*j/(2.0*N))
    return (c*x - s*y, s*x + c*y)


def butterfly((x, y), (N, j)):
    w = cmath.exp(-2.0j*math.pi*j/N)
    return (x + w*y, x - w*y)


def fft(v):
    n = len(v)
    if n == 1:
        return v
    evenfft = fft(v[0:][::2])
    oddfft = fft(v[1:][::2])
    result = [0]*n
    for k in xrange(n/2):
        (x, y) = butterfly((evenfft[k], oddfft[k]), (n, k))
        result[k] = x
        result[k + n/2] = y
    return result


def inverse_fft(v):
    def recursive_inverse_fft(v):
        n = len(v)
        if n == 1:
            return v
        evenfft = recursive_inverse_fft(v[0:][::2])
        oddfft = recursive_inverse_fft(v[1:][::2])
        result = [0]*n
        for k in xrange(n/2):
            (x, y) = butterfly((evenfft[k], oddfft[k]), (n, -k))
            result[k] = x
            result[k + n/2] = y
        return result
    return map(lambda x: x.real/len(v), recursive_inverse_fft(v))


def dct(v):
    n = len(v)
    t = fft((v + list(reversed(v)))[0:][::2])
    result = [0]*n
    for k in xrange(n/2):
        (x, y) = dct_butterfly((t[k].real, t[k].imag), (n, k))
        result[k] = x
        result[n - k - 1] = y
    return result
