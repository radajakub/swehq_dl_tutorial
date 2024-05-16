import scipy.stats
import torch

import math
import numpy as np

import matplotlib.pyplot as plt


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # these are needed for deepcopy / pickle
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


""" Simulation Model, similar to the one in the book 'The Elements of Statistical Learning' """


class DataModel:
    def __init__(self):
        np.random.seed(seed=1)
        self.K = 3  # mixture components
        self.priors = [0.5, 0.5]
        self.cls = [dotdict(), dotdict()]
        self.cls[0].mus = np.array([[-1, -1], [-1, 0], [0, 0]])
        self.cls[1].mus = np.array([[0, 1], [0, -1], [1, 0]])
        self.Sigma = np.eye(2) * 1 / 20
        self.name = 'GTmodel'

    def samples_from_class(self, c, sample_size):
        """
        :return: x -- [sample_size x d] -- samples from class c
        """
        # draw components
        kk = np.random.randint(0, self.K, size=sample_size)
        x = np.empty((sample_size, 2))
        for k in range(self.K):
            mask = kk == k
            # draw from Gaussian of component k
            x[mask, :] = np.random.multivariate_normal(self.cls[c].mus[k, :], self.Sigma, size=mask.sum())
        return x

    def generate_sample(self, sample_size):
        """
        function to draw labeled samples from the model
        :param sample_size: how many in total
        :return: (x,y) -- features, class, x: [sample_size x d],  y : [sample_size]
        """
        assert (sample_size % 2 == 0), 'use even sample size to obtain equal number of pints for each class'
        y = (np.arange(sample_size) >= sample_size // 2) * 1  # class labels
        x = np.zeros((sample_size, 2))
        for c in [0, 1]:
            # draw from Gaussian Mixture of class c
            x[y == c, :] = self.samples_from_class(c, sample_size // 2)
        y = 2 * y - 1  # remap to -1, 1
        return torch.tensor(x).to(torch.float32), torch.tensor(y)

    def score_class(self, c, x: np.array) -> np.array:
        """
            Compute log probability for data x and class c (sometimes also called score for the multinomial model)
            x: [N x d]
            return score : [N]
        """
        N = x.shape[0]
        S = np.empty((N, self.K))
        # compute log density of each mixture component
        for k in range(self.K):
            S[:, k] = scipy.stats.multivariate_normal(self.cls[c].mus[k, :], self.Sigma).logpdf(x)
        # compute log density of the mixture
        score = scipy.special.logsumexp(S, axis=1) + math.log(1.0 / self.K) + math.log(self.priors[c])
        return score

    def score(self, x: np.array) -> np.array:
        """ Return log odds (logits) of predictive probability p(y|x) of the network
        """
        scores = [self.score_class(c, x) for c in range(2)]
        score = scores[1] - scores[0]
        return score

    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for a given input
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class -1 or 1 per input point
        """
        return np.sign(self.score(x))

    def test_error(self, predictor, test_data):
        """
        evaluate test error of a predictor
        :param predictor: object with predictor.classify(x:np.array) -> np.array
        :param test_data: tuple (x,y) of the test points
        :return: error rate
        """
        x, y = test_data
        y1 = predictor.classify(x)
        err_rate = (y1 != y).sum() / x.shape[0]
        return err_rate

    def plot_boundary(self, train_data, predictor=None, title=None):
        """
        Visualizes the GT model, training points and the decisison boundary of a given predictor
        :param train_data: tuple (x,y)
        predictor: object with
            predictor.score(x:np.array) -> np.array
            predictor.name -- str to appear in the figure
        """
        x, y = train_data

        plt.figure(2)
        plt.rc('lines', linewidth=1)
        # plot points
        mask0 = y == -1
        mask1 = y == 1
        plt.plot(x[mask0, 0], x[mask0, 1], 'bo', ms=3)
        plt.plot(x[mask1, 0], x[mask1, 1], 'rd', ms=3)
        # plot classifier boundary
        ngrid = [200, 200]
        xx = [np.linspace(x[:, i].min() - 0.5, x[:, i].max() + 0.5, ngrid[i]) for i in range(2)]
        Xi, Yi = np.meshgrid(xx[0], xx[1], indexing='ij')  # 200 x 200 matrices
        X = np.stack([Xi.flatten(), Yi.flatten()], axis=1)  # 200*200 x 2
        score = self.score(X).reshape(ngrid)
        CS = plt.contour(Xi, Yi, score, [0], colors='r', linestyles='dashed')

        h, _ = CS.legend_elements()
        H = [h[0]]
        L = ["Bayes optimal"]
        if predictor is not None:
            X = torch.tensor(X).to(torch.float32)
            score = predictor(X).reshape(ngrid).detach().numpy()
            CS = plt.contour(Xi, Yi, score, [0], colors='k', linewidths=1)
            h, _ = CS.legend_elements()
            H += [h[0]]
            L += ["Predictor"]
            y1 = predictor.classify(x).squeeze()
            print(y1.shape)
            print(y.shape)
            err = y1 != y
            print(err.shape)
            h = plt.plot(x[err, 0], x[err, 1], 'ko', ms=6, fillstyle='none', label='errors', markeredgewidth=0.5)
            H += [h[0]]
            L += ["errors"]
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.legend(H, L, loc=0)
        if title is not None:
            plt.title(title)
