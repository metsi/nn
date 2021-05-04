import numpy as np

def monoset(d=3, s=.5, n=100, cc=[1,0,1]):
    cs = np.array([[-2*d], [-d], [d], [2*d]])
    X = np.array([np.random.normal(0, s, size=(n))+c
                  for c in cs])
    y = [np.ones(n)*c for c in cc]

    y = np.reshape(y, (-1))
    X = np.reshape(X, (-1, 1))

    return X, y


def dataset(d=3, s=.5, n=100, cc=[1,1,0,0]):
    cs = np.array([[-d,0], [0,-d], [d,0], [0,d]])
    X = np.array([np.random.normal(0, s, size=(n, 2))+c
                  for c in cs])
    y = [np.ones(n)*c for c in cc]

    y = np.reshape(y, (-1))
    X = np.reshape(X, (-1, 2))

    return X, y

def mspace(d=3, s=.5, q=50):
    xx = np.linspace(-(d+3*s), (d+3*s), q)

    return xx.reshape(-1, 1)


def mgrid(d=3, s=.5, q=50):
    xx = np.linspace(-(d+3*s), (d+3*s), q)
    XX, YY = np.meshgrid(xx, xx)

    X = np.array([XX.reshape(-1), YY.reshape(-1)]).T

    return X

def prob(d=3, s=.5, q=50):
    return np.linspace(-(d+3*s), (d+3*s), q)
