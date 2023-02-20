
import itertools
import numpy as np
from scipy.special import loggamma
from scipy.stats import beta



#@title Classical algorithms for multi-armed, linear, and GLM bandits
class UCB1:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.num_arms)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.num_arms)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.num_arms)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    ciw = self.crs * np.sqrt(2 * np.log(t))
    self.ucb = self.reward / self.pulls + \
      ciw * np.sqrt(1 / self.pulls) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB1"


class UCBV:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms
    self.n = n

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.num_arms)  # number of pulls
    self.reward = np.zeros(self.num_arms)  # cumulative reward
    self.reward2 = np.zeros(self.num_arms)  # cumulative squared reward
    self.tiebreak = 1e-6 * np.random.rand(self.num_arms)  # tie breaking

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r
    self.reward2[arm] += r * r

  def get_arm(self, t):
    if t < self.num_arms:
      # pull each arm once in the first K rounds
      self.ucb = np.zeros(self.num_arms)
      self.ucb[t] = 1
    else:
      # UCBs
      t += 1  # time starts at one

      # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / n^2) <= 1
      delta = 1.0 / np.power(self.n, 2)
      # # from \sum_{t = 1}^n \sum_{s = 1}^t (1 / t^3) <= \pi^2 / 6
      # delta = 1.0 / np.power(t, 3)

      muhat = self.reward / self.pulls
      varhat = (self.reward2 - self.pulls * np.square(muhat)) / self.pulls
      varhat = np.maximum(varhat, 0)
      self.ucb = muhat + \
        np.sqrt(2 * varhat * np.log(3 / delta) / self.pulls) + \
        3 * np.log(3 / delta) / self.pulls + \
        self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "UCB-V"


class KLUCB:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.num_arms)  # number of pulls
    self.reward = 1e-6 * np.random.rand(self.num_arms)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(self.num_arms)  # tie breaking

  def UCB(self, p, N, t):
    C = (np.log(t) + 3 * np.log(np.log(t) + 1e-6)) / N

    qmin = np.minimum(np.maximum(p, 1e-6), 1 - 1e-6)
    qmax = (1 - 1e-6) * np.ones(p.size)
    for i in range(16):
      q = (qmax + qmin) / 2
      ndx = (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) < C
      qmin[ndx] = q[ndx]
      qmax[~ndx] = q[~ndx]

    return q

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    # UCBs
    t += 1  # time starts at one
    self.ucb = \
      self.UCB(self.reward / self.pulls, self.pulls, t) + self.tiebreak

    arm = np.argmax(self.ucb)
    return arm

  @staticmethod
  def print():
    return "KL-UCB"


class TS:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms
    self.crs = 1.0  # confidence region scaling

    self.alpha = np.ones(self.num_arms)  # positive observations
    self.beta = np.ones(self.num_arms)  # negative observations

    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

  def update(self, t, arm, r):
    if (r > 0) and (r < 1):
      r = (np.random.rand() < r).astype(float)
    self.alpha[arm] += r
    self.beta[arm] += 1 - r

  def get_arm(self, t):
    if t < self.num_arms:
      # each arm is initially pulled once
      self.mu = np.zeros(self.num_arms)
      self.mu[t] = 1
    else:
      # posterior sampling
      crs2 = np.square(self.crs)
      self.mu = np.random.beta(self.alpha / crs2, self.beta / crs2)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "TS"


class GaussTS:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms
    self.sigma = 0.5

    self.mu0 = np.zeros(self.num_arms)
    self.sigma0 = 0.5 * np.ones(self.num_arms)

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = np.zeros(self.num_arms)  # number of pulls
    self.reward = np.zeros(self.num_arms)  # cumulative reward

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

  def get_arm(self, t):
    if t < self.num_arms:
      # each arm is initially pulled once
      self.mu = np.zeros(self.num_arms)
      self.mu[t] = 1
    else:
      # posterior distribution
      sigma2 = np.square(self.sigma)
      sigma02 = np.square(self.sigma0)
      post_var = 1.0 / (1.0 / sigma02 + self.pulls / sigma2)
      post_mean = post_var * (self.mu0 / sigma02 + self.reward / sigma2)

      # posterior sampling
      self.mu = post_mean + np.sqrt(post_var) * np.random.randn(self.num_arms)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Gaussian TS"


class EpsilonGreedy:
  def __init__(self, env, n, params):
    self.env = env
    self.num_arms = env.num_arms
    self.epsilon = self.num_arms / np.sqrt(n)
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    self.pulls = 1e-6 * np.ones(self.num_arms)  # number of pulls
    self.reward = 1e-6 * np.ones(self.num_arms)  # cumulative reward
    # self.tiebreak = 1e-6 * np.random.rand(self.num_arms)  # tie breaking

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.pulls[arm] += 1
    self.reward[arm] += r

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    # decision statistics
    muhat = self.reward / self.pulls
    best_arm = np.argmax(muhat)

    # probabilities of pulling arms
    eps = self.crs * self.epsilon
    p = (1 - eps) * (np.arange(self.num_arms) == best_arm) + eps / self.num_arms

    # pull the arm
    arm = best_arm
    if np.random.rand() < eps:
      arm = np.random.randint(self.num_arms)

    # derivative of the probability of the pulled arm
    self.grad[t] = self.epsilon * (1 / self.num_arms - (arm == best_arm)) / p[arm]

    return arm

  @staticmethod
  def print():
    return "e-greedy"


class Exp3:
  def __init__(self, env, n, params):
    self.env = env
    self.num_arms = env.num_arms
    self.crs = min(1, np.sqrt(self.num_arms * np.log(self.num_arms) / ((np.e - 1) * n)))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.eta = self.crs / self.num_arms
    self.reward = np.zeros(self.num_arms)  # cumulative reward

    self.grad = np.zeros(n)
    self.metrics = np.zeros((n, 3))

    # initialize baseline
    self.is_baseline = hasattr(self, "base_Alg")
    if self.is_baseline:
      self.base_alg = self.base_Alg(env, n, self.base_params)

  def update(self, t, arm, r):
    self.reward[arm] += r / self.phat[arm]

    best_r = self.env.rt[self.env.best_arm]
    if self.is_baseline:
      # baseline action and update
      base_arm = self.base_alg.get_arm(t)
      base_r = self.env.reward(base_arm)
      self.base_alg.update(t, base_arm, base_r)

      self.metrics[t, :] = np.asarray([r, r - best_r, r - base_r])
    else:
      self.metrics[t, :] = np.asarray([r, r - best_r, 0])

  def get_arm(self, t):
    # probabilities of pulling arms
    scaled_reward = self.reward - self.reward.max()
    p = np.exp(self.eta * scaled_reward)
    p /= p.sum()
    self.phat = (1 - self.crs) * p + self.eta

    # pull the arm
    q = np.cumsum(self.phat)
    arm = np.flatnonzero(np.random.rand() * q[-1] < q)[0]

    # derivative of the probability of the pulled arm
    self.grad[t] = (1 / self.phat[arm]) * \
      ((1 - self.crs) * (p[arm] / self.num_arms) *
      (scaled_reward[arm] - p.dot(scaled_reward)) - p[arm] + 1 / self.num_arms)

    return arm

  @staticmethod
  def print():
    return "Exp3"


class FPL:
  def __init__(self, env, n, params):
    self.num_arms = env.num_arms
    self.eta = np.sqrt((np.log(self.num_arms) + 1) / (self.num_arms * n))

    for attr, val in params.items():
      setattr(self, attr, val)

    self.loss = np.zeros(self.num_arms) # cumulative loss

  def update(self, t, arm, r):
    # estimate the probability of pulling the arm
    wait_time = 0
    while True:
      wait_time += 1
      ploss = self.loss + np.random.exponential(1 / self.eta, self.num_arms)
      if np.argmin(ploss) == arm:
        break;

    self.loss[arm] += (1 - r) * wait_time

  def get_arm(self, t):
    # perturb cumulative loss
    ploss = self.loss + np.random.exponential(1 / self.eta, self.num_arms)

    arm = np.argmin(ploss)
    return arm

  @staticmethod
  def print():
    return "FPL"


class LinBanditAlg:
  def __init__(self, env, n, params={}):
    self.env = env
    self.X = np.copy(env.X)
    self.num_arms = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.theta0 = np.zeros(self.d)
    self.sigma0 = 1.0
    self.sigma = 0.5
    self.crs = 1.0  # confidence region scaling

    for attr, val in params.items():
      setattr(self, attr, val)

    if not hasattr(self, "Sigma0"):
      self.Sigma0 = np.square(self.sigma0) * np.eye(self.d)

    self.pulls = np.zeros(self.num_arms)

    # sufficient statistics
    self.Gram = np.linalg.inv(self.Sigma0)
    self.B = self.Gram.dot(self.theta0)

    # posterior sample init
    self.mu = np.ones(self.num_arms)

  def update(self, t, arm, r):
    x = self.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)


class LinUCB(LinBanditAlg):
  def __init__(self, env, n, params={}):
    LinBanditAlg.__init__(self, env, n, params)

    self.cew = self.crs * self.confidence_ellipsoid_width(n)

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    delta = 1 / self.n
    L = np.amax(np.linalg.norm(self.X, axis=1))
    Lambda = 1 / np.square(self.sigma0)
    R = self.sigma
    S = np.sqrt(self.d)
    width = np.sqrt(Lambda) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
    return width

  def get_ucb(self):
    Gram_inv = np.linalg.inv(self.Gram)
    theta = Gram_inv.dot(self.B)

    # UCBs
    Gram_inv /= np.square(self.sigma)
    self.mu = self.X.dot(theta) + self.cew * \
              np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

  def get_arm(self, t):
    self.get_ucb()
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinUCB"


class LinGreedy(LinBanditAlg):
  def get_arm(self, t):
    self.mu = np.zeros(self.num_arms)
    if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.num_arms)] = np.Inf
    else:
      theta = np.linalg.solve(self.Gram, self.B)
      self.mu = self.X.dot(theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Lin e-greedy"

# from scipy.linalg import cholesky

class LinTS(LinBanditAlg):
  def update(self, t, arm, r):
    x = self.env.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_posterior(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)

    # posterior sampling
    cov_ = np.square(self.crs) * Gram_inv
    thetatilde = np.random.multivariate_normal(thetabar, cov_)  # unstable

    # Z_ = np.random.normal(size=(cov_.shape[0]))
    # thetatilde = Z_.dot(cholesky(cov_)) + thetabar

    self.mu = self.env.X.dot(thetatilde)
    return self.mu

  def get_arm(self, t):
    pos_sample = self.get_posterior(t)

    arm = np.argmax(pos_sample)
    self.pulls[arm] += 1
    return arm

  @staticmethod
  def print():
    return "LinTS"


class CoLinTS(LinBanditAlg):
  def update(self, t, arm, r):
    x = self.env.X[arm, :]
    self.Gram += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_arm(self, t):
    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      np.square(self.crs) * Gram_inv)
    self.mu = self.env.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    self.pulls[arm] += 1
    return arm

  @staticmethod
  def print():
    return "CoLinTS"


class LogBanditAlg:
  def __init__(self, env, n, params={}):
    self.env = env
    self.X = np.copy(env.X)
    self.num_arms = self.X.shape[0]
    self.d = self.X.shape[1]
    self.n = n
    self.sigma0 = 1.0
    self.crs = 1.0  # confidence region scaling
    self.crs_is_width = False

    self.irls_theta = np.zeros(self.d)
    self.irls_error = 1e-3
    self.irls_num_iter = 30

    self.pulls = np.zeros(self.num_arms)

    for attr, val in params.items():
      setattr(self, attr, val)

    # sufficient statistics
    self.pos = np.zeros(self.num_arms, dtype=int) # number of positive observations
    self.neg = np.zeros(self.num_arms, dtype=int) # number of negative observations
    self.X2 = np.zeros((self.num_arms, self.d, self.d)) # outer products of arm features
    for k in range(self.num_arms):
      self.X2[k, :, :] = np.outer(self.X[k, :], self.X[k, :])

  def update(self, t, arm, r):
    # import pdb; pdb.set_trace()
    self.pos[arm] += r
    self.neg[arm] += 1 - r

  def sigmoid(self, x):
    y = 1 / (1 + np.exp(- x))
    return y

  def solve(self):
    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      pulls = self.pos + self.neg
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        self.pos * (self.sigmoid(Xtheta) - 1) - \
        self.neg * (self.sigmoid(Xtheta) - 0)
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram


class LogUCB(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Section 4.1 in Filippi (2010)
    # Parametric Bandits: The Generalized Linear Case
    delta = 1 / self.n
    c_m = np.amax(np.linalg.norm(self.X, axis=1))
    c_mu = 0.25 # minimum derivative of the mean function
    k_mu = 0.25
    kappa = np.sqrt(3 + 2 * np.log(1 + 2 * np.square(c_m / self.sigma0)))
    R_max = 1.0
    width = (2 * k_mu * kappa * R_max / c_mu) * \
      np.sqrt(2 * self.d * np.log(t) * np.log(2 * self.d * self.n / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.sigmoid(self.X.dot(theta)) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-UCB (log)"


class UCBLog(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    if not self.crs_is_width:
      self.cew = self.crs * self.confidence_ellipsoid_width(n)
    else:
      self.cew = self.crs

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Li (2017)
    # Provably Optimal Algorithms for Generalized Linear Contextual Bandits
    delta = 1 / self.n
    sigma = 0.5
    kappa = 0.25 # minimum derivative of a constrained mean function
    width = (sigma / kappa) * \
      np.sqrt((self.d / 2) * np.log(1 + 2 * self.n / self.d) + \
      np.log(1 / delta))
    return width

  def get_arm(self, t):
    pulls = self.pos + self.neg
    Gram = np.tensordot(pulls, self.X2, axes=([0], [0])) + \
      np.eye(self.d) / np.square(self.sigma0)
    Gram_inv = np.linalg.inv(Gram)
    theta, _ = self.solve()

    # UCBs
    self.mu = self.X.dot(theta) + self.cew * \
      np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "UCB-GLM (log)"


class LogGreedy(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

    self.epsilon = 0.05

  def get_arm(self, t):
    self.mu = np.zeros(self.num_arms)
    if np.random.rand() < self.epsilon * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.num_arms)] = np.Inf
    else:
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Log e-greedy"


class LogTS(LogBanditAlg):
  def get_posterior(self, t, scale=1):
    thetabar, Gram = self.solve()
    Gram_inv = np.linalg.inv(Gram)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetabar,
      scale * np.square(self.crs) * Gram_inv)
    self.mu = self.X.dot(thetatilde)
    return
  
  def get_arm(self, t):
    self.get_posterior(t)  
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-TSL (log)"


class LogFPL(LogBanditAlg):
  def __init__(self, env, n, params):
    self.a = 1.0

    LogBanditAlg.__init__(self, env, n, params)

  def solve(self):
    # normal noise perturbation
    pulls = self.pos + self.neg
    z = self.a * np.sqrt(pulls) * \
      np.minimum(np.maximum(np.random.randn(self.num_arms), -6), 6)

    # iterative reweighted least squares for Bayesian logistic regression
    # Sections 4.3.3 and 4.5.1 in Bishop (2006)
    # Pattern Recognition and Machine Learning
    theta = np.copy(self.irls_theta)

    num_iter = 0
    while num_iter < self.irls_num_iter:
      theta_old = np.copy(theta)

      Xtheta = self.X.dot(theta)
      R = self.sigmoid(Xtheta) * (1 - self.sigmoid(Xtheta))
      Gram = np.tensordot(R * pulls, self.X2, axes=([0], [0])) + \
        np.eye(self.d) / np.square(self.sigma0)
      Rz = R * pulls * Xtheta - \
        (pulls * self.sigmoid(Xtheta) - (self.pos + z))
      theta = np.linalg.solve(Gram, self.X.T.dot(Rz))

      if np.linalg.norm(theta - theta_old) < self.irls_error:
        break;
      num_iter += 1

    if num_iter == self.irls_num_iter:
      self.irls_theta = np.zeros(self.d)
    else:
      self.irls_theta = np.copy(theta)

    return theta, Gram

  def get_arm(self, t):
    self.mu = np.zeros(self.num_arms)
    if t < self.d:
      self.mu[t] = np.Inf
    else:
      # history perturbation
      theta, _ = self.solve()
      self.mu = self.sigmoid(self.X.dot(theta))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "GLM-FPL (log)"


#@title Bandit simulator and environments
class BerBandit(object):
  """Bernoulli bandit environment."""

  def __init__(self, mu):
    self.mu = np.copy(mu)
    self.num_arms = self.mu.size

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand() < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Bernoulli bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class BetaBandit(object):
  """Beta bandit environment."""

  def __init__(self, mu, a_plus_b=4):
    self.mu = np.copy(mu)
    self.num_arms = self.mu.size
    self.a_plus_b = a_plus_b

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = \
      np.random.beta(self.a_plus_b * self.mu, self.a_plus_b * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Beta bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class GaussBandit(object):
  """Gaussian bandit environment."""

  def __init__(self, mu, sigma=0.5):
    self.mu = np.copy(mu)
    self.num_arms = self.mu.size
    self.sigma = sigma

    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = self.mu + self.sigma * np.random.randn(self.num_arms)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Gaussian bandit with arms (%s)" % \
      ", ".join("%.3f" % s for s in self.mu)


class LinBandit(object):
  """Linear bandit environment."""

  def __init__(self, X, theta, noise="normal", sigma=0.5):
    self.X = np.copy(X)
    self.num_arms = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)
    self.noise = noise
    if self.noise == "normal":
      self.sigma = sigma

    self.mu = self.X.dot(self.theta)
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    if self.noise == "normal":
      self.rt = self.mu + self.sigma * np.random.randn(self.num_arms)
    elif self.noise == "bernoulli":
      self.rt = (np.random.rand(self.num_arms) < self.mu).astype(float)
    elif self.noise == "beta":
      self.rt = np.random.beta(4 * self.mu, 4 * (1 - self.mu))

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    if self.noise == "normal":
      return "Linear bandit: %d dimensions, %d arms" % \
        (self.d, self.num_arms)
    elif self.noise == "bernoulli":
      return "Bernoulli linear bandit: %d dimensions, %d arms" % \
        (self.d, self.num_arms)
    elif self.noise == "beta":
      return "Beta linear bandit: %d dimensions, %d arms" % \
        (self.d, self.num_arms)


class LogBandit(object):
  """Logistic bandit."""

  def __init__(self, X, theta, **kw):
    # print("LogBandit")
    self.X = np.copy(X)
    self.num_arms = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)

    self.mu = 1 / (1 + np.exp(- self.X.dot(self.theta)))
    # print("LogBandit ", self.mu)
    self.best_arm = np.argmax(self.mu)

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = (np.random.rand(self.num_arms) < self.mu).astype(float)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Logistic bandit: %d dimensions, %d arms" % (self.d, self.num_arms)

  @staticmethod
  def ball_env(d=3, K=10, num_env=100):
    """Arm features and theta are generated randomly in a ball."""

    env = []
    for env_id in range(num_env):
      # standard d-dimensional basis (with a bias term)
      basis = np.eye(d)
      basis[:, -1] = 1

      # arm features in a unit (d - 2)-sphere
      X = np.random.randn(K, d - 1)
      X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
      X = np.hstack((X, np.ones((K, 1))))  # bias term
      X[: basis.shape[0], :] = basis

      # parameter vector in a (d - 2)-sphere with radius 1.5
      theta = np.random.randn(d - 1)
      theta *= 1.5 / np.sqrt(np.square(theta).sum())
      theta = np.append(theta, [0])

      # create environment
      env.append(LogBandit(X, theta))
      print("%3d: %.2f %.2f | " % (env[-1].best_arm,
        env[-1].mu.min(), env[-1].mu.max()), end="")
      if (env_id + 1) % 10 == 0:
        print()

    return env


class CoBandit(object):
  """Contextual bandit with linear generalization."""

  def __init__(self, X, Theta, sigma=0.5):
    self.X = np.copy(X)  # [number of contexs] x d feature matrix
    self.Theta = np.copy(Theta)  # d x [number of arms] parameter matrix
    self.num_arms = self.Theta.shape[1]
    self.d = self.X.shape[1]
    self.num_contexts = self.X.shape[0]
    self.sigma = sigma

    self.mu = self.X.dot(self.Theta)
    self.best_arm = np.argmax(self.mu, axis=1)

    self.randomize()

  def randomize(self):
    # choose context
    self.ct = np.random.randint(self.num_contexts)
    self.mut = self.mu[self.ct, :]

    # generate stochastic rewards
    self.rt = self.mut + self.sigma * np.random.randn(self.num_arms)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm[self.ct]] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mut[self.best_arm[self.ct]] - self.mut[arm]

  def print(self):
    return "Contextual bandit: %d dimensions, %d arms" % (self.d, self.num_arms)

