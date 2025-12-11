import numpy as np
import matplotlib.pyplot as plt
import math

class LinearReg:
    def __init__(self, iter=100, lern=0.00001, c=0, plot='False', track=False):
        # store hyperparameters inside the object
        self.iter = iter
        self.lern = lern
        self.c = c
        self.plot_mode = plot
        self.track = track

        # model parameters (will be set after fit)
        self.m = None
        self.fitted = False

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        n_samples, n_features = x.shape
        self.m = np.zeros((1, n_features))  # initialize weights

        # initial plot setup
        self._plot(x, y, yp=None, set_plot=True)

        for i in range(self.iter):
            yp = (self.m @ x.T).flatten() + self.c  # prediction

            # update weights
            for j in range(n_features):
                md = -(2 / n_samples) * np.sum(x[:, j] * (y - yp))
                self.m[0, j] -= self.lern * md
            cd = -(2 / n_samples) * np.sum(y - yp)
            self.c -= self.lern * cd

            # calculate loss
            mse = np.mean((y - yp) ** 2)

            if self.track:
                print(f"iter={i} c={self.c:.5f} m={self.m} cost={mse:.5f}")

            # update plots dynamically
            self._plot(x, y, yp=yp, set_plot=False)

        self.fitted = True

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        ypp = (self.m @ xt.T).flatten() + self.c
        return ypp

    def _plot(self, x, y, yp=None, set_plot=True):
        n_samples, n_features = x.shape

        if set_plot==True and self.plot_mode != False:
            if self.plot_mode == '*':
                cols = int(math.ceil(np.sqrt(n_features + 2)))
                rows = int(math.ceil((n_features + 2) / cols))
                self.fig, self.axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
                self.axes = np.array(self.axes).reshape(-1)

                # point size scaling
                def size(P, S_max=15, S_min=0.1):
                    return max(S_min, min(S_max, S_max / P))

                S = size(n_samples)

                for j in range(n_features):
                    self.axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features].set_title("Features vs y")

                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features + 1].set_title("yp vs y")

                plt.tight_layout()

            if self.plot_mode == True:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15.3, 7.8))
                self.axes = np.array(self.axes).reshape(-1)

                self.axes[0].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[0].set_title("Features vs y")

                self.axes[1].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[1].set_title("yp vs y")

                plt.tight_layout()

        if set_plot == False and self.plot_mode != False:
            if self.plot_mode == '*':
                for j in range(n_features):
                    self.axes[j].cla()
                    self.axes[j].scatter(x[:, j], y, c="blue", s=5)
                    self.axes[j].plot(x[:, j], self.m[0, j] * x[:, j] + self.c, c="red")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].cla()
                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[n_features].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[n_features].legend()

                self.axes[n_features + 1].cla()
                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[n_features + 1].plot(range(len(y)), yp, c="g")
                self.axes[n_features + 1].set_title("y vs yp")

                plt.pause(0.0001)

            if self.plot_mode == True:
                self.axes[0].cla()
                self.axes[0].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[0].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[0].legend()

                self.axes[1].cla()
                self.axes[1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[1].plot(range(len(y)), yp, c="g")
                self.axes[1].set_title("y vs yp")

                plt.pause(0.0001)

        if self.plot_mode != 'False':
            plt.show(block=False)


class Lasso:
    def __init__(self, iter=100, lern=0.00001,lamda=0.00001, c=0, plot='False', track=False):
        # store hyperparameters inside the object
        self.iter = iter
        self.lamda= lamda
        self.lern = lern
        self.c = c
        self.plot_mode = plot
        self.track = track

        # model parameters (will be set after fit)
        self.m = None
        self.fitted = False

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        n_samples, n_features = x.shape
        self.m = np.zeros((1, n_features))  # initialize weights

        # initial plot setup
        self._plot(x, y, yp=None, set_plot=True)

        for i in range(self.iter):
            yp = (self.m @ x.T).flatten() + self.c  # prediction

            # update weights
            for j in range(n_features):
                md = -(2 / n_samples) * np.sum(x[:, j] * (y - yp)) + (self.lamda * abs(self.m[:, j]))
                self.m[0, j] -= self.lern * md
            cd = -(2 / n_samples) * np.sum(y - yp)
            self.c -= self.lern * cd

            # calculate loss
            mse = np.mean((y - yp) ** 2)

            if self.track:
                print(f"iter={i} c={self.c:.5f} m={self.m} cost={mse:.5f}")

            # update plots dynamically
            self._plot(x, y, yp=yp, set_plot=False)

        self.fitted = True

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        ypp = (self.m @ xt.T).flatten() + self.c
        return ypp

    def _plot(self, x, y, yp=None, set_plot=True):
        n_samples, n_features = x.shape

        if set_plot==True and self.plot_mode != False:
            if self.plot_mode == '*':
                cols = int(math.ceil(np.sqrt(n_features + 2)))
                rows = int(math.ceil((n_features + 2) / cols))
                self.fig, self.axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
                self.axes = np.array(self.axes).reshape(-1)

                # point size scaling
                def size(P, S_max=15, S_min=0.1):
                    return max(S_min, min(S_max, S_max / P))

                S = size(n_samples)

                for j in range(n_features):
                    self.axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features].set_title("Features vs y")

                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features + 1].set_title("yp vs y")

                plt.tight_layout()

            if self.plot_mode == True:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15.3, 7.8))
                self.axes = np.array(self.axes).reshape(-1)

                self.axes[0].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[0].set_title("Features vs y")

                self.axes[1].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[1].set_title("yp vs y")

                plt.tight_layout()

        if set_plot == False and self.plot_mode != False:
            if self.plot_mode == '*':
                for j in range(n_features):
                    self.axes[j].cla()
                    self.axes[j].scatter(x[:, j], y, c="blue", s=5)
                    self.axes[j].plot(x[:, j], self.m[0, j] * x[:, j] + self.c, c="red")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].cla()
                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[n_features].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[n_features].legend()

                self.axes[n_features + 1].cla()
                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[n_features + 1].plot(range(len(y)), yp, c="g")
                self.axes[n_features + 1].set_title("y vs yp")

                plt.pause(0.0001)

            if self.plot_mode == True:
                self.axes[0].cla()
                self.axes[0].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[0].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[0].legend()

                self.axes[1].cla()
                self.axes[1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[1].plot(range(len(y)), yp, c="g")
                self.axes[1].set_title("y vs yp")

                plt.pause(0.0001)

        if self.plot_mode != 'False':
            plt.show(block=False)


class Ridge:
    def __init__(self, iter=100, lern=0.00001,lamda=0.00001, c=0, plot='False', track=False):
        # store hyperparameters inside the object
        self.iter = iter
        self.lamda= lamda
        self.lern = lern
        self.c = c
        self.plot_mode = plot
        self.track = track

        # model parameters (will be set after fit)
        self.m = None
        self.fitted = False

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        n_samples, n_features = x.shape
        self.m = np.zeros((1, n_features))  # initialize weights

        # initial plot setup
        self._plot(x, y, yp=None, set_plot=True)

        for i in range(self.iter):
            yp = (self.m @ x.T).flatten() + self.c  # prediction

            # update weights
            for j in range(n_features):
                md = -(2/n_samples)*np.sum(x[:, j] * (y-yp))+ (self.lamda * self.m[0,j] *self. m[0,j])
                self.m[0, j] -= self.lern * md
            cd = -(2 / n_samples) * np.sum(y - yp)
            self.c -= self.lern * cd

            # calculate loss
            mse = np.mean((y - yp) ** 2)

            if self.track:
                print(f"iter={i} c={self.c:.5f} m={self.m} cost={mse:.5f}")

            # update plots dynamically
            self._plot(x, y, yp=yp, set_plot=False)

        self.fitted = True

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        ypp = (self.m @ xt.T).flatten() + self.c
        return ypp

    def _plot(self, x, y, yp=None, set_plot=True):
        n_samples, n_features = x.shape

        if set_plot==True and self.plot_mode != False:
            if self.plot_mode == '*':
                cols = int(math.ceil(np.sqrt(n_features + 2)))
                rows = int(math.ceil((n_features + 2) / cols))
                self.fig, self.axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
                self.axes = np.array(self.axes).reshape(-1)

                # point size scaling
                def size(P, S_max=15, S_min=0.1):
                    return max(S_min, min(S_max, S_max / P))

                S = size(n_samples)

                for j in range(n_features):
                    self.axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features].set_title("Features vs y")

                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features + 1].set_title("yp vs y")

                plt.tight_layout()

            if self.plot_mode == True:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15.3, 7.8))
                self.axes = np.array(self.axes).reshape(-1)

                self.axes[0].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[0].set_title("Features vs y")

                self.axes[1].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[1].set_title("yp vs y")

                plt.tight_layout()

        if set_plot == False and self.plot_mode != False:
            if self.plot_mode == '*':
                for j in range(n_features):
                    self.axes[j].cla()
                    self.axes[j].scatter(x[:, j], y, c="blue", s=5)
                    self.axes[j].plot(x[:, j], self.m[0, j] * x[:, j] + self.c, c="red")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].cla()
                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[n_features].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[n_features].legend()

                self.axes[n_features + 1].cla()
                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[n_features + 1].plot(range(len(y)), yp, c="g")
                self.axes[n_features + 1].set_title("y vs yp")

                plt.pause(0.0001)

            if self.plot_mode == True:
                self.axes[0].cla()
                self.axes[0].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[0].plot(x[:, k], self.m[0, k] * x[:, k] + self.c, label=f"f{k + 1}")
                self.axes[0].legend()

                self.axes[1].cla()
                self.axes[1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[1].plot(range(len(y)), yp, c="g")
                self.axes[1].set_title("y vs yp")

                plt.pause(0.0001)

        if self.plot_mode != 'False':
            plt.show(block=False)


class PowerReg:
    def __init__(self, iter=100, lern=0.00001, c=0, plot='False', track=False):
        # store hyperparameters inside the object
        self.iter = iter
        self.lern = lern
        self.c = c
        self.plot_mode = plot
        self.track = track

        # model parameters (will be set after fit)
        self.m = None
        self.p = None
        self.fitted = False

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        n_samples, n_features = x.shape
        self.m = np.zeros((1, n_features))  # initialize weights
        self.p = np.ones((1, n_features))   # initialize powers

        # initial plot setup
        self._plot(x, y, yp=None, set_plot=True)

        for i in range(self.iter):
            yp = (self.m.flatten() * (x ** self.p.flatten())).sum(axis=1) + self.c  # prediction

            # update weights
            for j in range(n_features):
                md = -(2/n_samples)*np.sum(x[:, j] * (y-yp))
                self.m[0, j] -= self.lern * md
                pd = -(2 / n_samples) * np.sum((y - yp) * (self.m[0, j] * (x[:, j] ** self.p[0, j]) * np.log(x[:, j])))
                self.p[0, j] -= self.lern * pd
            cd = -(2 / n_samples) * np.sum(y - yp)
            self.c -= self.lern * cd

            # calculate loss
            mse = np.mean((y - yp) ** 2)

            if self.track:
                print(f"iter={i} c={self.c:.5f} m={self.m} cost={mse:.5f}")

            # update plots dynamically
            self._plot(x, y, yp=yp, set_plot=False)

        self.fitted = True

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        ypp = (self.m @ xt.T).flatten() + self.c
        return ypp

    def _plot(self, x, y, yp=None, set_plot=True):
        n_samples, n_features = x.shape

        if set_plot==True and self.plot_mode != False:
            if self.plot_mode == '*':
                cols = int(math.ceil(np.sqrt(n_features + 2)))
                rows = int(math.ceil((n_features + 2) / cols))
                self.fig, self.axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
                self.axes = np.array(self.axes).reshape(-1)

                # point size scaling
                def size(P, S_max=15, S_min=0.1):
                    return max(S_min, min(S_max, S_max / P))

                S = size(n_samples)

                for j in range(n_features):
                    self.axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features].set_title("Features vs y")

                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features + 1].set_title("yp vs y")

                plt.tight_layout()

            if self.plot_mode == True:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15.3, 7.8))
                self.axes = np.array(self.axes).reshape(-1)

                self.axes[0].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[0].set_title("Features vs y")

                self.axes[1].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[1].set_title("yp vs y")

                plt.tight_layout()

        if set_plot == False and self.plot_mode != False:
            if self.plot_mode == '*':
                for j in range(n_features):
                    self.axes[j].cla()
                    self.axes[j].scatter(x[:, j], y, c="blue", s=5)
                    self.axes[j].plot(x[:, j], (self.m[0, j]*(x[:, j]**self.p[0,j]))+self.c, c="red")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].cla()
                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[n_features].plot(x[:, k], (self.m[0, k]*(x[:, k]**self.p[0,k]))+self.c, label=f"f{k + 1}")
                self.axes[n_features].legend()

                self.axes[n_features + 1].cla()
                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[n_features + 1].plot(range(len(y)), yp, c="g")
                self.axes[n_features + 1].set_title("y vs yp")

                plt.pause(0.0001)

            if self.plot_mode == True:
                self.axes[0].cla()
                self.axes[0].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[0].plot(x[:, k], (self.m[0, k]*(x[:, k]**self.p[0,k]))+self.c, label=f"f{k + 1}")
                self.axes[0].legend()

                self.axes[1].cla()
                self.axes[1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[1].plot(range(len(y)), yp, c="g")
                self.axes[1].set_title("y vs yp")

                plt.pause(0.0001)

        if self.plot_mode != 'False':
            plt.show(block=False)


class LogesticReg:
    def __init__(self, iter=100, lern=0.0001,lamda=7.5, c=0, plot='False', track=False):
        # store hyperparameters inside the object
        self.iter = iter
        self.lamda= lamda
        self.lern = lern
        self.c = c
        self.plot_mode = plot
        self.track = track

        # model parameters (will be set after fit)
        self.m = None
        self.fitted = False

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        n_samples, n_features = x.shape
        self.m = np.zeros((1, n_features))  # initialize weights

        # initial plot setup
        self._plot(x, y, yp=None, set_plot=True)

        for i in range(self.iter):
            yp = 1 / (1 + np.exp((-self.m @ x.T).flatten() - self.c))  # prediction

            # update weights
            for j in range(n_features):
                md = -(2 / n_samples) * sum(((1 / yp) ** -2) * (y - yp) * (x[:, j] * np.exp((-self.m[:, j] * x[:, j]) - self.c)))
                self.m[0, j] -= self.lern * md
            cd = -(2 / n_samples) * sum(((1 / yp) ** -2) * (y - yp) * (np.exp(-(self.m @ x.T).flatten()) - self.c))
            self.c -= self.lamda * cd

            # calculate loss
            mse = np.mean((y - yp) ** 2)

            if self.track:
                print(f"iter={i} c={self.c:.5f} m={self.m} cost={mse:.5f}")

            # update plots dynamically
            self._plot(x, y, yp=yp, set_plot=False)

        self.fitted = True

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        ypp = (self.m @ xt.T).flatten() + self.c
        return ypp

    def _plot(self, x, y, yp=None, set_plot=True):
        n_samples, n_features = x.shape

        if set_plot==True and self.plot_mode != False:
            if self.plot_mode == '*':
                cols = int(math.ceil(np.sqrt(n_features + 2)))
                rows = int(math.ceil((n_features + 2) / cols))
                self.fig, self.axes = plt.subplots(rows, cols, figsize=(15.2, 9.4))
                self.axes = np.array(self.axes).reshape(-1)

                # point size scaling
                def size(P, S_max=15, S_min=0.1):
                    return max(S_min, min(S_max, S_max / P))

                S = size(n_samples)

                for j in range(n_features):
                    self.axes[j].scatter(x[:, j], y, c="blue", s=S, label="actual points")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features].set_title("Features vs y")

                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=S, label="actual points")
                self.axes[n_features + 1].set_title("yp vs y")

                plt.tight_layout()

            if self.plot_mode == True:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15.3, 7.8))
                self.axes = np.array(self.axes).reshape(-1)

                self.axes[0].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[0].set_title("Features vs y")

                self.axes[1].scatter(range(len(y)), y, c="blue", s=5, label="actual points")
                self.axes[1].set_title("yp vs y")

                plt.tight_layout()

        if set_plot == False and self.plot_mode != False:
            if self.plot_mode == '*':
                for j in range(n_features):
                    self.axes[j].cla()
                    self.axes[j].scatter(x[:, j], y, c="blue", s=5)
                    self.axes[j].plot(x[:, j], 1/(1+np.exp((-self.m[:,j]*x[:,j])-self.c)), c="red")
                    self.axes[j].set_title(f"Feature {j + 1} vs y")

                self.axes[n_features].cla()
                self.axes[n_features].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[n_features].plot(x[:, k], 1/(1+np.exp((-self.m[:,k]*x[:,k])-self.c)), label=f"f{k + 1}")
                self.axes[n_features].legend()

                self.axes[n_features + 1].cla()
                self.axes[n_features + 1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[n_features + 1].plot(range(len(y)), yp, c="g")
                self.axes[n_features + 1].set_title("y vs yp")

                plt.pause(0.0001)

            if self.plot_mode == True:
                self.axes[0].cla()
                self.axes[0].scatter(range(len(y)), y, c="blue", s=5)
                for k in range(n_features):
                    self.axes[0].plot(x[:, k], 1/(1+np.exp((-self.m[:,k]*x[:,k])-self.c)), label=f"f{k + 1}")
                self.axes[0].legend()

                self.axes[1].cla()
                self.axes[1].scatter(range(len(y)), y, c="blue", s=5)
                self.axes[1].plot(range(len(y)), yp, c="g")
                self.axes[1].set_title("y vs yp")

                plt.pause(0.0001)

        if self.plot_mode != 'False':
            plt.show(block=False)