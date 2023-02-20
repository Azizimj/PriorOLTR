"""Main source code for the Algorithms and Environments

"""

from scipy.stats import beta
import numpy as np

# utils
def get_top_k(pos_sample, k_list):
  return np.argsort(pos_sample)[: : -1][: k_list]

def misspecify(beta_a, beta_b, misspecification):
  return beta_a + misspecification, beta_b - misspecification

#@title Environments
#%%
class SpikeEnv:
  """Environment where the examination probabilities and
     attractiveness/relevance probabilities are uniform except for a single
     item. The first item in the slate has higher probability of examination
     than the rest, and only one item in the set of all items has
     higher probability of being attractive/relevant.
  """
  def __init__(self, k_list = 4, L = 16, baseu = 0.5, gapu = 0.4, basev = 0.5, gapv = 0.4):
    self.k_list = k_list  # Slate size
    self.L = L  # Total number of items

    self.ubar = baseu * np.ones(k_list)  # examination probs for slate items
    self.ubar[0] += gapu  # first item more likely to be examined
    self.vbar = basev * np.ones(L)  # attraction/relevance probs for all items
    self.vbar[L // 2] += gapv  # the middle item is more likely to be relevant
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]  # the optimal action is to choose the k_list most attractive items

    self.ut = np.zeros(k_list)  # actual attraction/relevance (binary)
    self.vt = np.zeros(L)  # actual examination (binary)

  def name():
    return 'Spike'

  def randomize(self):
    # sample random variables
    self.ut = np.array(np.random.rand(self.k_list) < self.ubar, dtype = int)  
    self.vt = np.array(np.random.rand(self.L) < self.vbar, dtype = int)

  def reward(self, action):
    # reward of action (chosen items)
    return np.multiply(self.ut, self.vt[action])

  def reward_scalar(self, action):
    return np.dot(self.ut, self.vt[action])

  def regret(self, action):
    # regret of action (chosen items)
    return self.reward_scalar(self.best_action) - self.reward_scalar(action)

  def expected_reward(self, action):
    # expected regret of action (chosen items)
    return np.dot(self.ubar, self.vbar[action])

  def pregret(self, action):
    # expected regret of action (chosen items)
    return self.expected_reward(self.best_action) - self.expected_reward(action)

  def plot(self):
    # plot model parameters
    fig, (left, right) = plt.subplots(ncols = 2, figsize = (10, 3))
    left.plot(self.ubar)
    right.plot(self.vbar)
    # plt.show()

class PBMEnv(SpikeEnv):
  def __init__(self):
    self.k_list = 5
    self.L = 10

    self.ubar = np.random.rand(self.k_list)  # random examination probs
    self.vbar = np.random.rand(self.L)  # random attraction/relevance probs
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]

    self.ut = np.zeros(self.k_list)
    self.vt = np.zeros(self.L)

  def name():
    return 'PBM'


class CMEnv:
  def __init__(self):
    # print("CMEnv")
    self.k_list = 5
    self.L = 10

    self.vbar = np.random.rand(self.L)  # random attraction/relevance
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]

    self.vt = np.zeros(self.L)

  def name():
    return 'CM'

  def randomize(self):
    # sample random variables
    self.vt = np.array(np.random.rand(self.L) < self.vbar, dtype=int)  # realize attraction/relevance

  def reward(self, action):
    # reward of action (chosen items)
    r = self.vt[action]
    if r.sum() > 0:
      first_click = np.flatnonzero(r)[0]
      r[first_click + 1 :] = 0
    return r

  def reward_scalar(self, action):
    return 1 - np.prod(1 - self.reward(action))
    # return np.sum(self.reward(action))

  def regret(self, action):
    # regret of action (chosen items)
    return self.reward_scalar(self.best_action) - self.reward_scalar(action)

  def expected_reward(self, action):
    return 1 - np.prod(1 - self.vbar[action])

  def pregret(self, action):
    # expected regret of action (chosen items)
    return self.expected_reward(self.best_action) - self.expected_reward(action)


class DCMEnv:
  def __init__(self, k_list, L, beta_a, beta_b):
    self.k_list = k_list  # Slate size
    self.L = L  # Total number of items

    if (len(beta_a) != self.L  or
        len(beta_b) != self.L):
      raise ValueError("Prior params dimension mismatch.")
    self.beta_a = beta_a
    self.beta_b = beta_b

    # Sample problem instance from prior
    self.vbar = np.random.beta(self.beta_a, self.beta_b)
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]
    self.p_exit_given_click = 0.5 * np.ones(self.k_list)  # exit probabilities

    self.vt = np.zeros(self.L)
    self.et = np.zeros(self.k_list)

  def name():
    return 'DCM'

  def get_prior(self):
    return (self.beta_a, self.beta_b)

  def randomize(self):
    # sample random variables
    self.vt = np.array(np.random.rand(self.L) < self.vbar,
                       dtype=int)  # realize attraction/relevance
    self.et = np.array(np.random.rand(self.k_list) < self.p_exit_given_click,
                       dtype=int)

  def reward(self, action):
    # reward of action (chosen items)
    r = self.vt[action]
    if (r * self.et).sum() > 0:
      # noting after click that is exiting
      first_click = np.flatnonzero(r * self.et)[0]
      r[first_click + 1 :] = 0
    return r

  def reward_scalar(self, action):
    return 1 - np.prod(1 - (self.reward(action) * self.et))

  def regret(self, action):
    # regret of action (chosen items)
    return self.reward_scalar(self.best_action) - self.reward_scalar(action)

  def expected_reward(self, action):
    return 1 - np.prod(1 - self.vbar[action] * self.p_exit_given_click)

  def pregret(self, action):
    # expected regret of action (chosen items)
    return self.expected_reward(self.best_action) - self.expected_reward(action)


#%% Envoronment with Prior
class PriorCMEnv(CMEnv):
  """Environment defined by a Beta prior for each item (arm). Using a cascade
     model for examination.
  """
  def __init__(self, k_list, L, beta_a, beta_b, **kw):
    # print("PriorCMEnv")
    self.k_list = k_list  # Slate size
    self.L = L  # Total number of items

    if (len(beta_a) != self.L  or
        len(beta_b) != self.L):
      raise ValueError("Prior params dimension mismatch.")
    self.beta_a = beta_a
    self.beta_b = beta_b

    # Sample problem instance from prior
    self.vbar = np.random.beta(self.beta_a, self.beta_b)
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]

    self.vt = np.zeros(self.L)

    self.name = 'CM'

  def name():
    return 'PriorCM'

  def get_prior(self):
    return (self.beta_a, self.beta_b)


#%%
class PriorDCTREnv(PriorCMEnv):
  def name():
    return 'PriorDCTR'

  def reward(self, action):
    # reward of action (chosen items)
    r = self.vt[action]
    return r

  def reward_scalar(self, action):
    return np.sum(self.reward(action))

  def expected_reward(self, action):
    return np.sum(self.vbar[action])


#%% Web30k Envoronment
class LTREnv(CMEnv):
  """Environment for a learning-to-rank query (datsets).
  """

  def __init__(self, beta_a, beta_b, click_probs,
               scores_sum=[], scores_sq=[], k_list=5):
    self.k_list = k_list
    self.L = len(click_probs)
    if (len(beta_a) != self.L or
        len(beta_b) != self.L):
      raise ValueError("Prior params dimension mismatch.")
    self.beta_a = beta_a
    self.beta_b = beta_b
    self.vbar = np.asarray(click_probs)
    self.scores_sum = scores_sum
    self.scores_sq = scores_sq
    self.best_action = np.argsort(self.vbar)[: : -1][: self.k_list]

    self.vt = np.zeros(self.L)

  def get_prior(self):
    return (self.beta_a, self.beta_b)

  def get_scores(self):
    return (np.array(self.beta_a)+np.array(self.beta_b),
            self.scores_sum, self.scores_sq)  # num model, score stats

  def name():
    return "LTR"

#@ title Algorithms
#%%
class BayesianBeta:
  def __init__(self, L, T):
    self.L = L
    self.T = T  # should not use this if the algo doesn't know it.

    self.pulls = np.zeros(L)
    self.reward = np.zeros(L)

    self.rewards = [list() for _ in range(L)] # only used in fMeta_TS DEPREC

    # Need to call set_prior to set these:
    self.beta_a = np.zeros(L)
    self.beta_b = np.zeros(L)

  def name():
    raise NotImplementedError()

  def set_prior(self, beta_a, beta_b):
    if (len(beta_a) != len(self.beta_a)  or
        len(beta_b) != len(self.beta_b)):
      raise ValueError("Prior params dimension should match.")
    self.beta_a = beta_a
    self.beta_b = beta_b

  def update(self, t, action, r):
    """
    t: time
    action: list of item index
    r: list of attraction 0,1
    """
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r
    for cnt, act in enumerate(action):
      self.rewards[act].append(r[cnt])

  def posterior_params(self):
    a = self.beta_a + self.reward
    b = self.beta_b + (self.pulls - self.reward)
    return a, b


class ThompsonSampling(BayesianBeta):
  def name():
    return "ThompsonSampling"

  def posterior_sample(self):
    a, b = self.posterior_params()
    return np.random.beta(a, b)

  def get_action(self, t, k_list):
    posterior_sample = self.posterior_sample()
    action = get_top_k(posterior_sample, k_list)
    return action


class BayesUCB(BayesianBeta):
  def name():
    return "BayesUCB"

  def get_action(self, t, k_list):
    # Compute posterior
    a, b = self.posterior_params()
    q = 1. - 1. / (t+1)  # quantile
    ucbs = beta.ppf(q, a, b)
    action = get_top_k(ucbs, k_list)
    return action


#%%
class ThompsonSamplingPriorOnly(ThompsonSampling):
  def name():
    return "ThompsonSampling (prior only)"

  def posterior_params(self):
    return self.beta_a, self.beta_b


class BayesUCBPriorOnly(BayesUCB):
  def name():
    return "BayesUCB (prior only)"

  def posterior_params(self):
    return self.beta_a, self.beta_b


#%%
class Greedy():
  def __init__(self, L, T):
    self.L = L

    self.pulls = np.zeros(L)
    self.reward = np.zeros(L)

  def name():
    return "Greedy"

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r

  def get_action(self, t, k_list):
    posterior_means = self.reward / (self.pulls + 1)  # prevent dividsion by 0
    action = get_top_k(posterior_means, k_list)
    return action


class Ensemble(BayesianBeta):
  def name():
    return "Ensemble"

  def get_action(self, t, K):
    action = np.argsort(self.beta_a)[: : -1][: K]
    return action

#%%
class CascadeUCB1:
  def __init__(self, k_list, T):
    self.k_list = k_list

    self.pulls = 1e-6 * np.ones(k_list)  # number of pulls
    self.reward = 1e-6 * np.random.rand(k_list)  # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(k_list)  # tie breaking

    self.ucb = np.zeros(k_list)

  def name():
    return "CascadeUCB1"

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r

    # UCBs
    t += 1  # time starts at one
    ct = np.maximum(np.sqrt(2 * np.log(t)), 2)
    self.ucb = self.reward / self.pulls + ct * np.sqrt(1 / self.pulls)

  def get_action(self, t, num_pulls):
    action = np.argsort(self.ucb + self.tiebreak)[: : -1][: num_pulls]
    return action


class CascadeKLUCB:
  def __init__(self, k_list, T):
    self.k_list = k_list

    self.pulls = 1e-6 * np.ones(k_list) # number of pulls
    self.reward = 1e-6 * np.random.rand(k_list) # cumulative reward
    self.tiebreak = 1e-6 * np.random.rand(k_list) # tie breaking

    self.ucb = (1 - 1e-6) * np.ones(k_list)

  def name():
    return "CascadeKLUCB"

  def UCB(self, p, N, t):
    C = (np.log(t) + 3 * np.log(np.log(t) + 1)) / N
    tol = 1e-5

    kl = p * np.log(p / self.ucb) + (1 - p) * np.log((1 - p) / (1 - self.ucb))
    for k in np.flatnonzero(np.abs(kl - C) > tol):
      ql = min(max(p[k], 1e-6), 1 - 1e-6)
      qu = 1 - 1e-6
      while qu - ql > tol:
        q = (ql + qu) / 2
        f = p[k] * np.log(p[k] / q) + (1 - p[k]) * np.log((1 - p[k]) / (1 - q))
        if f < C[k]:
          ql = q
        else:
          qu = q
      self.ucb[k] = qu

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r

    # UCBs
    t += 1 # time starts at one
    self.UCB(self.reward / self.pulls, self.pulls, t)

  def get_action(self, t, num_pulls):
    action = np.argsort(self.ucb + self.tiebreak)[: : -1][: num_pulls]
    return action


class TopRank:
  def __init__(self, k_list, T):
    self.k_list = k_list
    self.T = T

    self.pulls = np.ones((k_list, k_list)) # number of pulls
    self.reward = np.zeros((k_list, k_list)) # cumulative reward

    self.G = np.ones((k_list, k_list), dtype = bool)
    self.P = np.zeros(k_list)
    self.P2 = np.ones((k_list, k_list))

  def name():
    return "TopRank"

  def rerank(self):
    Gt = (self.reward / self.pulls - 2 * np.sqrt(np.log(self.T) / self.pulls)) > 0
    if not np.array_equal(Gt, self.G):
      self.G = np.copy(Gt)

      Pid = 0
      self.P = - np.ones(self.k_list)
      while (self.P == -1).sum() > 0:
        items = np.flatnonzero(Gt.sum(axis = 0) == 0)
        self.P[items] = Pid
        Gt[items, :] = 0
        Gt[:, items] = 1
        Pid += 1

      self.P2 = \
        (np.tile(self.P[np.newaxis], (self.k_list, 1)) == np.tile(self.P[np.newaxis].T, (1, self.k_list))).astype(float)

  def update(self, t, action, r):
    clicks = np.zeros(self.k_list)
    clicks[action] = r

    M = np.outer(clicks, 1 - clicks) * self.P2
    self.pulls += M + M.T
    self.reward += M - M.T

    self.rerank()

  def get_action(self, t, num_pulls):
    action = np.argsort(self.P + 1e-6 * np.random.rand(self.k_list))[: num_pulls]
    return action


#@title Gaussian LTR
def beta_mean_var(beta_a, beta_b):
  _mean = beta_a/(beta_a+beta_b)
  _var = (beta_a*beta_a) / ((beta_a+beta_a)**2 * (beta_a+beta_a+1))
  return _mean, np.sqrt(_var)
  

class BayesianGaussBeta:
  def __init__(self, L, T):
    self.L = L

    self.pulls = np.zeros(L)
    self.reward = np.zeros(L)

    # Need to call set_prior to set these:
    self.beta_a = np.zeros(L)
    # self.beta_b = np.zeros(num_items)

    self.sigma2 = 1/4  #  reward var

    self.mu0 = 0
    self.mu0 = 0.5
    self.sigma02 = 1
    self.sigma02 = 1/4

  def name():
    raise NotImplementedError()

  def set_prior(self, beta_a, beta_b):
    raise NotImplementedError()

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    self.pulls[action] += 1
    self.reward[action] += r


class GaussTS_LTR(BayesianGaussBeta):

  def name():
    return "GaussTS_LTR"

  def get_action(self, t, k_list):
    Ti = self.pulls
    ybar = self.reward / np.maximum(1, Ti)
    
    posterior_sample = np.random.normal(ybar, 1/np.maximum(1, Ti))
    
    # sig_tilde = 1/self.sigma02 + Ti/self.sigma2
    # mu_tilde = (self.mu0/self.sigma02 + self.reward/self.sigma2)/sig_tilde
    # sig_tilde = np.sqrt(1/sig_tilde)
    # posterior_sample = np.random.normal(mu_tilde, sig_tilde)
    
    action = np.argsort(posterior_sample)[:: -1][: k_list]
    return action

  def set_prior(self, mu0, sig0):
    self.mu0, self.sigma02 = mu0, sig0


class GPTS(BayesianGaussBeta):

  def name():
    return "GPTS"

  def set_scores(self, num_model, scores_sum, scores_sq):
    if (len(scores_sum) != len(self.beta_a)  or
        len(scores_sq) != len(self.beta_a)):
      raise ValueError("Prior params dimension should match.")
    num_model = np.maximum(1, num_model)
    self.mu0 = scores_sum /num_model
    self.sigma02 = np.maximum(scores_sq/num_model - self.mu0**2, 0)
    # print(self.mu0, self.sigma02, self.mu0.shape, self.sigma02.shape)

  def set_prior(self, beta_a, beta_b):
    if (len(beta_a) != len(self.beta_a)  or
        len(beta_b) != len(self.beta_a)):
      raise ValueError("Prior params dimension should match.")
    beta_a, beta_b = np.array(beta_a), np.array(beta_b)
    self.mu0 = beta_a/(beta_a+beta_b)
    self.sigma02 = (beta_a*beta_a) / ((beta_a+beta_a)**2 * (beta_a+beta_a+1))

  def get_action(self, t, k_list):

    Ti = self.pulls
    ybar = self.reward / np.maximum(1, Ti)
    sig_tilde = 1/self.sigma02 + Ti/self.sigma2
    mu_tilde = (self.mu0/self.sigma02 + self.reward/self.sigma2)/sig_tilde
    sig_tilde = np.sqrt(1/sig_tilde)
    # sig_tilde = 1/sig_tilde

    # posterior_sample = mu_tilde + sig_tilde*np.random.randn(self.num_items)
    posterior_sample = np.random.normal(mu_tilde, sig_tilde)
    # posterior_sample = np.random.normal(mu_tilde, 1/(Ti+1))
    # posterior_sample = np.random.normal(ybar, 1/(Ti+1)) #
    action = np.argsort(posterior_sample)[:: -1][: k_list]
    return action


class GPTSmean(GPTS):

  def name():
    return "GPTSmean"
  
  def get_action(self, t, k_list):
    Ti = self.pulls
    ybar = self.reward / np.maximum(1, Ti)
    sig_tilde = 1/self.sigma02 + Ti/self.sigma2
    mu_tilde = (self.mu0/self.sigma02 + self.reward/self.sigma2)/sig_tilde

    # posterior_sample = np.random.normal(mu_tilde, np.sqrt(1/(Ti+1)))
    posterior_sample = np.random.normal(mu_tilde, 1/(Ti+1))
    action = np.argsort(posterior_sample)[:: -1][: k_list]
    return action


class TS_Cascade(BayesianGaussBeta):

  def name():
    return "TS_Cascade"

  def get_action(self, t, k_list):
    Ti = self.pulls
    ybar = self.reward / np.maximum(1, Ti)
    vhat = ybar*(1-ybar)
    sighat = np.maximum(np.sqrt(vhat*np.log(t+1)/(Ti+1)), np.log(t+1)/(Ti+1))

    posterior_sample = ybar + np.random.randn(1)*sighat
    # posterior_sample = np.random.normal(ybar, sighat)

    action = np.argsort(posterior_sample)[:: -1][: k_list]
    return action



#@title Contextual LTR
from context_bandits_lib import LinBandit, LinTS


#%% Contextual LTR Environments
class LTREnv_features(LTREnv):
  """Environment for a learning-to-rank query (datasets) with features.
  """
  def __init__(self, X, **kwargs):
    self.X = X
    super().__init__(**kwargs)

  def name():
    return "LTREnv_features"


class LinBandit_LTR_Env(src_cb.LinBandit):
  """Linear bandit LTR environment."""

  def __init__(self, k_list, L, beta_a=[], beta_b=[], preds=[], **kwargs):
    self.k_list = k_list
    self.L = L
    self.num_arms = self.L
    super().__init__(**kwargs)
    self.best_action = get_top_k(self.mu, k_list)

    # if len(preds) == 0:  # for the synthetic case
    #   preds = self.mu
    # # import pdb; pdb.set_trace()  # Blaze debug
    # prob_click = expit(np.asarray(preds))
    # self.beta_a = np.maximum(prob_click, 1)
    # self.beta_b = np.maximum(10-prob_click, 1)

    if len(beta_a) == 0:
      beta_a, beta_b = self.L * [1], self.L * [1]
    self.beta_a = beta_a
    self.beta_b = beta_b
    # self.click_probs = np.asarray(click_probs)

  def name():
    return "LinBandit_LTR_Env"

  def get_prior(self):
    return (self.beta_a, self.beta_b)

  def reward_scalar(self, action):
    return np.sum(self.reward(action))

  def regret(self, action):
    # regret of action (chosen items)
    return self.reward_scalar(self.best_action) - self.reward_scalar(action)

  def expected_reward(self, action):
    return np.sum(self.mu[action])

  def pregret(self, action):
    # expected regret of action (chosen items)
    return self.expected_reward(self.best_action) - self.expected_reward(action)


class LinBandit_LTR_Env_DEP(PriorCMEnv, src_cb.LinBandit):
  """Linear bandit LTR environment. DEPRECATED"""
  def __init__(self, k_list, X, theta, beta_a, beta_b,
               click_probs=[], sigma=.25):
    self.k_list = k_list
    L = np.asarray(X).shape[0]
    self.num_arms = L
    self.L = L
    PriorCMEnv.__init__(self, k_list=k_list, L=L, beta_a=beta_a, beta_b=beta_b)
    src_cb.LinBandit.__init__(self, X=X, theta=theta, sigma=sigma)

    self.vbar = self.mu  # use the logistic mean from LogBandit
    if len(click_probs) > 0:  # for dataset
      self.vbar = click_probs
    self.best_action = get_top_k(self.mu, k_list)
    # import pdb; pdb.set_trace()

  def name():
    return "LogBandit_LTR_Env"


class LogBandit_LTR_Env(PriorCMEnv, src_cb.LogBandit):
  """Logistic bandit LTR environment."""
  def __init__(self, k_list, X, theta, beta_a, beta_b,
      click_probs=[]):
    self.k_list = k_list
    L = np.asarray(X).shape[0]
    self.num_arms = L
    self.L = L
    PriorCMEnv.__init__(self, k_list=k_list, L=L, beta_a=beta_a, beta_b=beta_b)
    src_cb.LogBandit.__init__(self, X, theta)
    # super().__init__(k_list=k_list, num_items=num_items, X=X, theta=theta,
    #                  beta_a=beta_a, beta_b=beta_b
    #                  )  # LogBandit does not get called!
    # import pdb; pdb.set_trace()

    self.vbar = self.mu  # use the logistic mean from LogBandit
    if len(click_probs) > 0:  # for dataset
      self.vbar = click_probs
    self.best_action = get_top_k(self.mu, k_list)
    # import pdb; pdb.set_trace()

  def name():
    return "LogBandit_LTR_Env"

"""Contextual LTR Algorithms"""
##############################
from scipy.linalg import sqrtm

class LinTS_LTR(src_cb.LinTS):
  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]

    for cnt, _item in enumerate(action):
      super().update(t, _item, r[cnt])

  def posterior_cov(self, t):
    lambda_ = 1e-4
    S_ = 1
    delta_ = 1/(self.n*(np.log(self.n)+2))
    delta_t = delta_/2**max(1, np.ceil(np.log(t+1)))
    beta_t = self.sigma**2 * lambda_ *\
             np.sqrt(2*np.log((lambda_+t+1)**(self.d/2)
                              /delta_t/lambda_**(self.d/2))) \
             +np.sqrt(lambda_) * S_
    # print("LinTS_LTR", "beta_t", beta_t)
    return beta_t

  def get_posterior(self, t):
    # import pdb; pdb.set_trace()  # Blaze debug

    Gram_inv = np.linalg.inv(self.Gram)
    thetabar = Gram_inv.dot(self.B)
    # posterior sampling
    cov_ = np.square(self.crs) * Gram_inv
    # beta_t = self.posterior_cov(t)
    beta_t = 1
    xit = np.random.randn(self.d)
    thetatilde = thetabar + beta_t * sqrtm(cov_).dot(xit)
    self.mu = self.env.X.dot(thetatilde)
    return self.mu

  def get_action(self, t, k_list):
    # print("LinTS_LTR get_action")
    pos_sample = self.get_posterior(t)

    arm = get_top_k(pos_sample, k_list)
    self.pulls[arm] += 1
    return arm

  def name():
    return "LinTS_LTR"


class LinTS_Cascade(LinTS_LTR):

  def __init__(self, env, T):
    self.k_list = env.k_list
    # self.num_arms = env.num_items
    super().__init__(env, T)

  def posterior_cov(self, t):
    lambda_ = 1e-4
    R_ = 1/4
    vt_ = 3 * R_ * np.sqrt(self.d * np.log(t+1))
    beta_t = lambda_ * vt_ * np.sqrt(self.k_list)
    # print("LinTS_Cascade", "beta_t", beta_t)
    return beta_t

  def name():
    return "LinTS_Cascade"


class Cascade_LinUCB(src_cb.LinUCB):
  """Zong et al. 2016.

  https://arxiv.org/abs/1603.05359
  """
  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]
    for cnt, _item in enumerate(action):
      super().update(t, _item, r[cnt])
      # x = self.env.X[_item, :]
      # self.Gram += np.outer(x, x) / np.square(self.sigma)
      # self.B += x * r[cnt] / np.square(self.sigma)

  def get_action(self, t, k_list):
    # print("LinTS_LTR get_action")
    self.get_ucb()
    mu_ = np.minimum(self.mu, 1)
    arm = get_top_k(mu_, k_list)
    self.pulls[arm] += 1
    return arm

  def name():
    return "Cascade_LinUCB"


class CascadeWOFUL(Cascade_LinUCB):
  """Vial et al. 2022.

  https://arxiv.org/abs/2203.12577
  """
  def __init__(self, env, T):
    self.k_list = env.k_list
    super().__init__(env, T)

    # Bernstein statistics
    self.Gram_Bernstein = np.linalg.inv(self.Sigma0)
    self.B_Bernstein = self.Gram.dot(self.theta0)

  def update(self, t, action, r):
    super().update(t, action, r)
    self.get_ucb()  # to update self.mu to be used here.
    mu_ = np.maximum(self.mu, 1/self.k_list)
    for cnt, _item in enumerate(action):
      x = self.env.X[_item, :]
      self.Gram_Bernstein += np.outer(x, x) / np.square(self.sigma)/mu_[_item]
      self.B_Bernstein += x * r[cnt] / np.square(self.sigma)/mu_[_item]

  def get_Bernstein_ucb(self):
    Gram_inv = np.linalg.inv(self.Gram_Bernstein)
    theta = Gram_inv.dot(self.B_Bernstein)

    # UCBs
    Gram_inv /= np.square(self.sigma)
    self.mu_Bernstein = self.X.dot(theta) + self.k_list * self.cew * \
              np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis=1))

  def get_action(self, t, k_list):
    self.get_Bernstein_ucb()
    arm = get_top_k(self.mu_Bernstein, k_list)
    self.pulls[arm] += 1
    return arm

  def name():
    return "CascadeWOFUL"


class LogTS_LTR(src_cb.LogTS):
  # (TODO): beat standard: do we need to overwrite update like in LinTS_LTR?

  def get_action(self, t, k_list):
    # self.get_posterior(t, np.sqrt(2 * np.log(t+1)))
    # self.get_posterior(t, k_list/(t+1))
    # self.get_posterior(t, 1/(t+1))
    self.get_posterior(t)

    arm = get_top_k(self.mu, k_list)
    self.pulls[arm] += 1
    return arm

  def update(self, t, action, r):
    if r.sum() > 0:
      last_click = np.flatnonzero(r)[-1]
      action = action[: last_click + 1]
      r = r[: last_click + 1]
    for cnt, _item in enumerate(action):
      super().update(t, _item, r[cnt])

  def name():
    return "LogTS_LTR"



LINEAR_ALGS = [LinTS_LTR,
               LinTS_Cascade,
               Cascade_LinUCB,
               CascadeWOFUL]  # the list of linear algorithms
LOG_ALGS = [LogTS_LTR]

#@title colors and styles, etc
alg_colors = {
    BayesUCB.name(): "orange",
    ThompsonSampling.name(): "green",
    Greedy.name(): "purple",
    CascadeUCB1.name(): "gray",
    CascadeKLUCB.name(): "blue",
    TopRank.name(): "red",
    TS_Cascade.name(): "plum",
    LinTS_Cascade.name(): "yellowgreen",
    Cascade_LinUCB.name(): "grey",
    CascadeWOFUL.name(): "silver",
    Ensemble.name(): "dimgrey",

    # ours
    GaussTS_LTR.name(): "blue",
    "GaussTS": "blue",
    GPTS.name(): "blue",
    GPTSmean.name(): "blue",
    LinTS_LTR.name(): "cyan",
    LogTS_LTR.name(): "royalblue",
    # GaussTS_LTR.name(): "gold",
    # LinTS_LTR.name(): "olive",
    # LogTS_LTR.name(): "plum",
    fMeta_TS.name(): "cyan",
    fMeta_TS_explore.name(): "royalblue",

}

alg_labels = {
    # BayesUCB.name(): "BayesUCB-Beta",
    # ThompsonSampling.name(): "TS-Beta",
    # CascadeUCB1.name(): "CascadeUCB1",
    # CascadeKLUCB.name(): "CascadeKL-UCB",
    # TopRank.name(): "TopRank",
    # Greedy.name(): "Greedy",
    # fMeta_TS.name(): "fMeta-TS",
    # fMeta_TS_explore.name(): "fMeta-TS-explore",
    # GaussTS_LTR.name(): "GTS (ours)",
    # GPTS.name(): "GTS-P",
    # GPTSmean.name(): "GTS-Pmean",
    # TS_Cascade.name(): "TS-Cascade",
    # LinTS_LTR.name(): "LinTS-LTR (ours)",
    # LinTS_Cascade.name(): "LinTS-Cascade",
    # LogTS_LTR.name(): "LogTS-LTR (ours)",
    # Cascade_LinUCB.name(): "Cascade_LinUCB",
    # CascadeWOFUL.name(): "CascadeWOFUL",
    # Ensemble.name(): "Ensemble"

    BayesUCB.name(): "BayesUCB-Beta",
    ThompsonSampling.name(): "TS-Beta",
    CascadeUCB1.name(): "CascadeUCB1",
    CascadeKLUCB.name(): "CascadeKL-UCB",
    TopRank.name(): "TopRank",
    Greedy.name(): "Greedy",
      fMeta_TS.name(): "fMeta-TS",
      fMeta_TS_explore.name(): "fMeta-TS-explore",
    GaussTS_LTR.name(): "GTS (ours)",
                 "GaussTS":"GTS (ours)",
    GPTS.name(): "GTS-P",
    GPTSmean.name(): "GTS-Pmean",
    TS_Cascade.name(): "TS-Cascade",
    LinTS_LTR.name(): "LinTS-LTR (ours)",
    LinTS_Cascade.name(): "LinTS-Cascade",
    LogTS_LTR.name(): "LogTS-LTR (ours)",
    Cascade_LinUCB.name(): "C-LinUCB",
    CascadeWOFUL.name(): "C-WOFUL",
    Ensemble.name(): "Ensemble"
}

alg_line_styles = {
    BayesUCB.name(): None,
    ThompsonSampling.name(): "--",

    GaussTS_LTR.name(): None,
    "GaussTS": None,
    LinTS_LTR.name(): 'dashdot',
    LogTS_LTR.name(): ":",

    TS_Cascade.name(): None,
    LinTS_Cascade.name(): "--",

    Cascade_LinUCB.name(): None,
    CascadeWOFUL.name(): ":",

    Ensemble.name(): None,

    Greedy.name(): None, GPTS.name(): "-.",
    GPTSmean.name(): ":", CascadeUCB1.name(): None,
    CascadeKLUCB.name(): None, TopRank.name(): None,
    fMeta_TS.name(): ':', fMeta_TS_explore.name(): '-.',
    ThompsonSamplingPriorOnly.name(): "--",
}

env_styles = {
    PriorCMEnv.name(): '-',
    PriorDCTREnv.name(): '--',
    DCMEnv.name(): ':'
}

# distributions names
GAUSS = "Gaussian"; BERN = "Bernoulli"
# experiment types
linear_ex_type = "Lin-exp"
stand_ex_type = "Standard-exp"
log_ex_type = "Log-exp"

# environment types
LTREnv_type = LTREnv.name()
LTREnv_features_type = LTREnv_features.name()
LinBandit_LTR_Env_type = LinBandit_LTR_Env.name()
LogBandit_LTR_Env_type = LogBandit_LTR_Env.name()

# @title Eval
import copy
def evaluate_one(run_num, _Bandit, env, T, period_size=1, random_seed=110,
    misspecification=0, ppos_flg=0, GTS_prior=None, _verbose=False):
  if _verbose:
    print("run_num", run_num)

  if _Bandit in (LINEAR_ALGS + LOG_ALGS):
    bandit = _Bandit(env, T)
  else:
    bandit = _Bandit(env.L, T)
  np.random.seed(random_seed)

  if isinstance(bandit, BayesianBeta):
    prior_func = getattr(env, "get_prior", None)
    if callable(prior_func):
      beta_a, beta_b = prior_func()
      if misspecification:
        beta_a, beta_b = misspecify(beta_a, beta_b, misspecification)
      bandit.set_prior(beta_a, beta_b)
    else:
      raise ValueError("Can only use Thompson Sampling with environments that "
                       "have a prior.")

  if GTS_prior and (_Bandit in [GaussTS_LTR]):
    prior_func = getattr(env, "get_prior", None)
    if callable(prior_func):
      beta_a, beta_b = prior_func()
      beta_m, beta_sd = beta_mean_var(beta_a, beta_b)
    _GTS_prior = copy.copy(GTS_prior)
    if GTS_prior[0] == "BetaMean":
      if GTS_prior[1] == "BetaSd":
        _GTS_prior = [beta_m, beta_sd]
      else:
        _GTS_prior[0] = beta_m
    bandit.set_prior(mu0=_GTS_prior[0], sig0=_GTS_prior[1])

  if _Bandit in [GPTS, GPTSmean]:
    prior_score_fnc = getattr(env, "get_scores", None)
    prior_func = getattr(env, "get_prior", None)
    if callable(prior_score_fnc):
      num_model, scores_sum, scores_sq = prior_score_fnc()
      bandit.set_scores(num_model, scores_sum, scores_sq)
    elif callable(prior_func):
      beta_a, beta_b = prior_func()
      bandit.set_prior(beta_a, beta_b)
    else:
      raise ValueError("Can only use GPTS with environments that "
                       "have a get_scores or get_prior.")

  # if _Bandit in [LogTS_LTR]: # TODO give LogTS the prior

  # print(_Bandit.name(), "run_num", run_num,
  #       "best_action: ", env.best_action)

  regret = np.zeros(T // period_size)
  learned_prior_pos = []
  for t in range(T):
    # generate state
    env.randomize()

    # take action
    action = bandit.get_action(t, env.k_list)

    # update model and regret
    bandit.update(t, action, env.reward(action))
    regret_t = env.regret(action)
    regret[t // period_size] += regret_t
    # if t % 100 == 0:
    #   print(t, _Bandit.name(), action, regret_t)
    # if regret_t > 0:
    #   print(t, _Bandit.name(), action, "regret_t", regret_t
    #         #, "inter", set(action).intersection(set(env.best_action))
    #         )

    if ppos_flg:
      ahat, bhat = bandit.posterior_params()
      if ppos_flg==2:
        if not (np.any(ahat==0) or np.any(bhat==0)):
          a, b = env.get_prior()
          learned_prior_pos.append(np.linalg.norm(a / (a + b) - ahat / (ahat + bhat)))
        else:
          learned_prior_pos.append(-.1)
      elif ppos_flg==1:
        if not (np.any(np.isnan(ahat)) or np.any(np.isnan(bhat))):
          a, b = env.get_prior()
          tmp = (np.linalg.norm(ahat - a), np.linalg.norm(bhat - b))
          learned_prior_pos.append(tmp)
        else:
          learned_prior_pos.append(-.1)

  return (regret, bandit, np.array(learned_prior_pos))


def evaluate(_Bandit, env, num_exps=5, T=1000, period_size=1,
    display=False, misspecification=0, reload_=False, fname=None,
    GTS_prior=None, plot_pos=0, parr=0, write_save=0, load_res=None,
    num_cpu=None, mp=None, save_res=None, _verbose=True):
  """Evaluate an algorithm on an instantiated (fixed) env (frequentist regret)."""
  if reload_:
    regret = load_res(fname)
    return (regret, None, None, None)

  if display:
    print("Simulation with %s positions and %s items" % (env.k_list, env.L))

  ppos_flg = _Bandit.name() in [fMeta_TS.name(), ThompsonSampling.name(), 
                                BayesUCB.name()]
  ppos_flg = ppos_flg * plot_pos

  seeds = np.random.randint(2 ** 15 - 1, size=num_exps)
  if parr:
    print(f"MP running {num_cpu}!")
    output = []
    def collect_result(res):
      output.append(res)
    pool = mp.Pool(num_cpu)
    poolobjs = [pool.apply_async(
        evaluate_one, args=[run_num, _Bandit, env, T, period_size,
                            seeds[run_num], misspecification, ppos_flg,
                            GTS_prior, _verbose],
        callback=collect_result) for run_num in range(num_exps)]
    print("before close")
    pool.close()
    # for f in poolobjs:
    #   print(f.get())  # print the errors
    pool.join()
    # for f in poolobjs:
    #   print(f.get())  # print the errors
  else:
    output = [evaluate_one(ex, _Bandit, env, T, period_size, seeds[ex],
                           misspecification, ppos_flg,
                           GTS_prior, _verbose) for ex in range(num_exps)]

  regret = np.vstack([item[0] for item in output]).T
  bandit = output[-1][1]
  # learned_prior_pos = np.vstack([item[2][:,0] for item in output]).T
  learned_prior_pos_mean, learned_prior_pos_sd = [], []
  if ppos_flg:
    learned_prior_pos_mean = np.mean([item[2] for item in output], axis=0)
    learned_prior_pos_sd = np.std([item[2] for item in output], axis=0)

  if display:
    regretT = np.sum(regret, axis = 0)
    print("Regret: %.2f \\pm %.2f, " % (np.mean(regretT), np.std(regretT) / np.sqrt(num_exps)))

  if (not reload_) and write_save:
    save_res(fname=fname, res=regret)
    # save_res(fname=f"expr{exp_name}-{_Bandit.name()}-{env_name}-{time_stamp}", res=[1])

  # import pdb; pdb.set_trace()
  return (regret,  # shape: T * num_exps
          bandit, learned_prior_pos_mean, learned_prior_pos_sd)




