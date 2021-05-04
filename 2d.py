import numpy as np
import matplotlib.pyplot as plt
import methods as m
from sklearn.neural_network import MLPClassifier

# Probing parameters
q = 32
X, y = m.dataset(cc=[1,0,1,0])
X_test = m.mgrid(q = q, d = 10)

# Network parameters
epochs = 100
n_neurons = 1000
hls = (n_neurons,)
conf = {
#    'activation': 'identity',
#    'solver': 'sgd',
#    'learning_rate_init': .1,
#    'random_state': 1411
}
clf = MLPClassifier(hls, **conf).partial_fit(X, y, [0,1])

imgs = []
imgs_p = []

for epoch in range(epochs):
    if epoch >=3:
        clf.partial_fit(X, y)
    y_pred = clf.predict(X_test)
    y_pp = clf.predict_proba(X_test)[:,0]

    fig, ax = plt.subplots(3,2,figsize=(10,15))
    ax[0,0].scatter(X[:,0], X[:,1], c=y, cmap = 'bwr')
    ax[1,0].scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap = 'binary')
    ax[1,0].scatter(X[:,0], X[:,1], c=y, cmap = 'bwr')
    ax[0,0].set_title(epoch)

    img = y_pred.reshape(q, q)
    img_p = y_pp.reshape(q, q)

    imgs.append(img)
    imgs_p.append(img_p)

    if len(imgs) >= 3:
        imgg = 255-(np.array(imgs[-3:]).swapaxes(0,2)*255).swapaxes(0,1)
        imgg_p = (np.array(imgs_p[-3:]).swapaxes(0,2)).swapaxes(0,1)

        ax[0,1].imshow(imgg_p, origin='lower')
        ax[1,1].imshow(imgg, origin='lower')

    ax[2,1].hist(clf.predict_proba(X_test)[:,0], 32)

    w = clf.coefs_
    b = clf.intercepts_
    nimg = np.concatenate(
        (
            w[0], w[1].T, [b[0]]
        )
    )

    ax[2,0].set_title(b[1])
    ax[2,0].imshow(nimg)

    plt.tight_layout()
    plt.savefig("foo.png")
    plt.savefig("frames/%04i.png" % epoch)
    plt.close('all')
