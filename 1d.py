import numpy as np
import matplotlib.pyplot as plt
import methods as m
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as metric
import matplotlib


epochs = 100
q = 32
X, y = m.monoset(cc=[1,1,0,0])
X_test = m.mspace(q = q, d = 5)

n_neurons = 1
hls = (n_neurons)
conf = {
    'activation': 'identity',
    'solver': 'sgd',
    'learning_rate_init': .1,
#    'random_state': 1411
}
clf = MLPClassifier(hls, **conf).partial_fit(X, y, [0,1])

#clf.coefs_[0] = np.random.normal(size=clf.coefs_[0].shape)
#clf.coefs_[1] = np.random.normal(size=clf.coefs_[1].shape)
#clf.intercepts_[0] = clf.intercepts_[0] * 0
#clf.intercepts_[1] = clf.intercepts_[1] * 0

y_pps = []
y_preds = []
scores = []

for epoch in range(epochs):
    y_pred = clf.predict(X_test)
    y_pp = clf.predict_proba(X_test)[:,0]

    y_pps.append(y_pp)
    y_preds.append(y_pred)

    score = metric(y, clf.predict(X))

    scores.append(score)

    if epoch >= 3:
        fig, ax = plt.subplots(3,2,figsize=(10,10))

        ax[0,0].set_title(epoch)

        colors = 1-np.array(y_preds[-3:]).T
        ax[0,0].scatter(X_test[:,0],
                        X_test[:,0], edgecolor='black', c=colors)
        ax[0,0].scatter(X[:,0], X[:,0]*0, c=y, cmap = 'binary', edgecolor='black')


        y_ppss = np.array(y_pps[-3:]).T

        ax[0,1].plot(X_test, np.array(y_pps[-5:]).T, c = 'red')
        ax[1,1].plot(X_test, np.array(y_pps[-1:]).T, c = 'red')
        ax[0,1].plot(X_test, np.array(y_pps[-1:]).T, c = 'black')
        ax[0,1].plot(X_test, [.5 for _ in y_ppss], ls=":", c='black')
        ax[0,1].set_ylim(0,1)

        # Plot neurons
        w = clf.coefs_
        b = clf.intercepts_

        print(w[0])
        print(b[0])

        for n_idx in range(n_neurons):
            a_ = w[0][0, n_idx]
            b_ = b[0][n_idx]

            ax[1,1].plot(X_test, X_test*a_ + b_, c='blue', ls=":")

            if n_idx > 10:
                break

        a_ = w[1][0, 0]
        b_ = b[1][0]

        ax[1,0].plot(scores)
        ax[1,1].plot(X_test, X_test*a_ + b_, c='green')
        ax[1,0].set_ylim(0,1.01)
        ax[1,1].set_title(hls)

        # NIMG
        nimg = np.concatenate(
            (
                w[0], w[1].T, [b[0]]
            )
        )

        print(nimg.shape)

        ax[2,0].set_title(b[1])
        ax[2,0].imshow(nimg)

        plt.tight_layout()
        plt.savefig("foo.png")
        plt.savefig("frames/%04i.png" % epoch)
        plt.close('all')

        clf.partial_fit(X, y)
