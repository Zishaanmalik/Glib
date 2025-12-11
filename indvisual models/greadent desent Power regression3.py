import numpy as np
import matplotlib.pyplot as plt
import math


def PowerReg(x, y, iter=100, lern=0.0000001, c=0, plot='False',track=True):
    # check if input is np array or not
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    n_samples, n_features = x.shape
    m = np.zeros((1, n_features))
    p = np.ones((1, n_features))

    # plot all the graph
    if plot == '*':
        # calculate rows and rows
        # n = 5:
        # cols = floor(√5) = 2
        # rows = ceil(5 / 2) = 3
        # = 2 × 3
        # grid

        # n = 7:
        # cols = floor(√7) = 2
        # rows = ceil(7 / 2) = 4
        # = 2 × 4
        # grid(morehorizontal, looksnatural)

        cols = int(math.ceil(np.sqrt(n_features + 2)))
        rows = int(math.ceil((n_features + 2) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
        axes = np.array(axes).reshape(-1)  # flatten for easy indexing

        # adjudt size of points accourding to lenth of data
        def size(P, S_max=15, S_min=0.1, P_max=1000):
            K = S_max
            S = K / P
            S = max(S_min, min(S_max, S))
            return S

        S = size(n_samples)

        for j in range(n_features):
            axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
            axes[j].set_title(f"Feature {j + 1} vs y")
            axes[j].set_xlabel(f"Feature {j + 1}")
            axes[j].set_ylabel("y")

        # all subplot: combined all features
        axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
        axes[n_features].set_title("Features vs y")
        axes[n_features].set_xlabel("Samples")
        axes[n_features].set_ylabel("y")

        # last subplot: combined all features
        axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
        axes[n_features + 1].set_title("yp vs y")
        axes[n_features + 1].set_xlabel("Samples")
        axes[n_features + 1].set_ylabel("yp")

        plt.tight_layout()

    if plot == True:
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(15.3, 7.8))
        axes = np.array(axes).reshape(-1)  # flatten for easy indexing

        # adjudt size of points accourding to lenth of data
        def size(P, S_max=15, S_min=0.1, P_max=1000):
            K = S_max
            S = K / P
            S = max(S_min, min(S_max, S))
            return S

        S = size(n_samples)

        # all subplot: combined all features
        axes[0].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
        axes[0].set_title("Features vs y")

        # last subplot: combined all features
        axes[1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
        axes[1].set_title("yp vs y")

        plt.tight_layout()

    for i in range(iter):
        yp = (m.flatten() * (x ** p.flatten())).sum(axis=1) + c  # prediction
        # update weights
        for j in range(n_features):
            md = -(2 / n_samples) * np.sum(x[:, j] * (y - yp))
            m[0, j] -= lern * md
            pd = -(2 / n_samples) * np.sum((y - yp) * (m[0, j] * (x[:, j]**p[0, j]) * np.log(x[:, j])))
            p[0, j] -= lern * pd
        cd = -(2 / n_samples) * np.sum(y - yp)
        c -= lern * cd

        # calculate loss
        mse = np.mean((y - yp) ** 2)

        # traking data
        if (track == True):
            print(f"iter={i} c={c:.5f} p:{p} m={m} cost={mse:.5f}")

        # update all plots dynamically
        if plot == '*':
            for j in range(n_features):
                axes[j].cla()
                axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                axes[j].plot(x[:, j], (m[0, j]*(x[:, j]**p[0,j])) + c, c="red", label="predicted line")
                axes[j].set_title(f"Feature {j + 1} vs y")
                axes[j].legend()

            # combined plot
            axes[n_features].cla()
            axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
            for k in range(n_features):
                axes[n_features].plot(x[:, k], (m[0, k] * (x[:, k] ** p[0, k])) + c, label=f"f{k + 1}")
            axes[n_features].set_title("All features vs y")
            axes[n_features].legend()

            # final plot
            axes[n_features + 1].cla()
            axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
            axes[n_features + 1].plot(range(len(y)), yp, c='g', label="yp")
            axes[n_features + 1].set_title("y vs yp")
            axes[n_features + 1].legend()

            plt.tight_layout()
            plt.pause(0.000001)

        # update 2 plots dynamically
        if plot == True:
            # combined plot
            axes[0].cla()
            axes[0].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
            for k in range(n_features):
                axes[0].plot(x[:, k], (m[0, k]*(x[:, k]**p[0,k])) + c, label=f"f{k+1}")
            axes[0].set_title("All features vs y")
            axes[0].set_xlabel("Samples")
            axes[0].set_ylabel("y")
            axes[0].legend()

            # final plot
            axes[1].cla()
            axes[1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
            axes[1].plot(range(len(y)), yp, c='g', label="yp")
            axes[1].set_title("y vs yp")
            axes[1].set_xlabel("Samples")
            axes[1].set_ylabel("y")
            axes[1].legend()

            plt.tight_layout()
            plt.pause(0.000001)

    if plot != 'False':
        plt.show()

# Example usage
x = [[i*1.111, i, i*1.03, i*1.15, i*0.6, i*1.09, i*1.1] for i in range(1,50)]  # 4 features
y = [i**2 for i in range(1,50)]

PowerReg(x, y, iter=100, plot=True)

