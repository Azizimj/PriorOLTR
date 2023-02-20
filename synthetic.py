# -*- coding: utf-8 -*-
"""Local runner for bandits
"""

#@title import
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle

from absl import app

import multiprocessing as mp
from collections import defaultdict

#@title Global Setup
parr = 1  # use `mp` or not
parr = 0
write_save = 0  # save results for reload (DEPREC)
PICKLE_SAVE = True  # save results for reload
PICKLE_LOAD = False  # load results for reload
plot_pos = 0  # 1 plot posterior params, 2 plot a/(a+b) diff norm

SAVE_PLOT = True  # true for saving the pdf plots.

from ranking_bandits_lib import *
from utils import make_dir, get_pst_time
PROJ_SAVE_DIR = "./"

num_cpu = min(mp.cpu_count(), 10)

FIGS_DIR = './ltr_figs/'
RES_DIR = './LTRresult/'


"""plt Setup"""
plt.tight_layout();
np.random.seed(110);

mpl.rcParams["figure.figsize"] = [5, 3];

mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["legend.handlelength"] = 3
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = 3
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

"""End Setup"""
#@title Evaluation utils

def save_fig(fig_name):
  fig_file = FIGS_DIR + fig_name
  print("saving ", fig_file)
  plt.tight_layout()
  if SAVE_PLOT:
    plt.savefig(fig_file, format="pdf", bbox_inches = 0)
  plt.show()


def save_res(fname, res):
  print("Saving", fname)
  with open(PROJ_SAVE_DIR + RES_DIR + fname+".pkl", "wb") as f:
      pickle.dump(res, f)
  print(f"Saved {fname}")


def load_res(filename):
  print("Loading", filename)
  with open(PROJ_SAVE_DIR + RES_DIR + filename+".pkl", "rb") as f:
    tmp = pickle.load(f)
    return tmp


## plotting utils
def plot_prior_pos(ppos, time_stamp, L, k_list, T, reload_, exp_name, tt):
  """
  tt: time to cut at
  """
  if reload_:
    ppos = {kk: {'mean': None, 'sd': None} for kk in alg_colors}

  _line_styles = ['-', '--', '.-']
  for alg in ppos:
    clr = alg_colors[alg]
    if len(ppos[alg]["mean"]) == 0:
      continue
    if (not reload_) and write_save:
      save_res(fname=f"exp{exp_name}-ppos-{alg}-L{L}-K{k_list}-horizon{T}-tt{tt}-{time_stamp}-mean",res=ppos[alg]['mean'][:tt])
      save_res(fname=f"exp{exp_name}-ppos-{alg}-L{L}-K{k_list}-horizon{T}-tt{tt}-{time_stamp}-sd",res=ppos[alg]['sd'][:tt])
    if reload_:
      ppos[alg]['mean'] = load_res(fname=f"exp{exp_name}-ppos-{alg}-L{L}-K{k_list}-horizon{T}-tt{tt}-{time_stamp}-mean")
      ppos[alg]['sd'] = load_res(fname=f"exp{exp_name}-ppos-{alg}-L{L}-K{k_list}-horizon{T}-tt{tt}-{time_stamp}-sd")

    if plot_pos==1:
      for i in range(ppos[alg]['mean'].shape[1]):
        plt.fill_between(range(tt),
                         ppos[alg]['mean'][:,i][:tt] + ppos[alg]['sd'][:,i][:tt],
                         ppos[alg]['mean'][:,i][:tt] - ppos[alg]['sd'][:,i][:tt],
                         alpha=0.1, edgecolor="none", facecolor=clr)
        plt.plot(ppos[alg]['mean'][:,i][:tt], color=clr, label=alg+str(i), linestyle=_line_styles[i], linewidth=3)
    elif plot_pos==2:
      plt.fill_between(range(T), ppos[alg]['mean'][:tt] + ppos[alg]['sd'][:tt],
                     ppos[alg]['mean'][:tt] - ppos[alg]['sd'][:tt],
                     alpha=0.1, edgecolor="none", facecolor=clr)
      plt.plot(ppos[alg]['mean'][:tt], color=clr, label=alg, linestyle=alg_line_styles[alg], linewidth=3)

  plt.legend(loc='upper left')
  plt.xlabel("Steps")
  if plot_pos==1:
    plt.ylabel("Prior misspec norm")
  elif plot_pos==2:
    plt.ylabel("Prior mean misspec norm")
  plt.title("L={}, k_list={}".format(L, k_list))
  fig_file = FIGS_DIR + f"exp{exp_name}-ppos-L{L}-K{k_list}-horizon{T}-tt{tt}-{time_stamp}.pdf"
  plt.tight_layout()
  # plt.savefig(fig_file, format="pdf", bbox_inches=0)
  save_fig(fig_file)
  plt.close()


def plot_exper(T_plots, algs, mean_sds, num_items, k_list, expr_fname,
    plot_pos=False, pos_prior=[], time_stamp="", horizon=1, reload_=False,
    period_size=1):
  # import pdb; pdb.set_trace()
  for tt in T_plots:
    steps = range(tt)[::period_size]
    tt = tt//period_size
    for alg in algs:
      alg_name = alg.name()
      me_sd = mean_sds[alg_name]
      plt.fill_between(steps,
                       me_sd[0][:tt] + 3*me_sd[1][:tt],
                       me_sd[0][:tt] - 3*me_sd[1][:tt],
                       alpha=0.1, edgecolor="none", facecolor=alg_colors[alg_name])
      plt.plot(me_sd[0][:tt], color=alg_colors[alg_name], label=alg_labels[alg_name],
               linestyle=alg_line_styles[alg_name], linewidth=3)
      print(f"Final regret", alg_name, tt, "\t\t mean", me_sd[0][tt-1],
            "sd", me_sd[1][tt-1])

    plt.legend(loc='upper left', prop={'weight':'bold', 'size':10}, markerscale=5)
    plt.xlabel("Round", fontweight='bold', fontsize=10)
    plt.ylabel("Cumulative Regret", fontweight='bold', fontsize=10)
    plt.tight_layout()

    fig_file = expr_fname + f"-tt{tt}.pdf"
    save_fig(fig_file)
    plt.close()

    if plot_pos:
      plot_prior_pos(pos_prior, time_stamp=time_stamp, L=num_items, k_list=k_list,
                     T=horizon, reload_=reload_, exp_name=1, tt=tt)

## experiment utils
def regret_to_cumregret(regret, num_runs):
  cum_regret = np.cumsum(regret, axis=0)
  mean_vals = cum_regret.mean(axis=1)  # size: [horizon]
  se_vals = cum_regret.std(axis=1) / np.sqrt(num_runs)  # size: [horizon]
  return mean_vals, se_vals


def eval_algs_env(alg, env, time_stamp, num_runs, exp_name, horizon, parr, period_size=1):
  fname = f"{exp_name}-{alg.name()}-{time_stamp}"
  regret, _, _, _ = evaluate(alg, env, num_runs, horizon,
                             reload_=reload_, fname=fname
                             , plot_pos=plot_pos, parr=parr
                             , write_save=write_save, period_size=period_size,
                             load_res=load_res, num_cpu=num_cpu, mp=mp,
                             save_res=save_res, _verbose=True)
  return regret_to_cumregret(regret, num_runs)


def eval_algs_env_bayes(run_num, algs, ex_type, k_list, num_items, dim,
    horizon, period_size=1, random_seed=110, env_type=None, reset=False, parr=0):
  run_mean_sds = defaultdict(list)
  env = init_env(k_list=k_list, num_items=num_items, dim=dim,
                 ex_type=ex_type, random_seed=random_seed,
                 env_type=env_type, reset=reset)
  for alg in algs:
    # regret = evaluate_one(run_num, alg, env, horizon, period_size, random_seed)[0]
    # run_mean_sds[alg.name()].append(regret)
    mean_regret, _ = eval_algs_env(alg=alg, env=env, time_stamp=-1, num_runs=10,
                           exp_name=None, horizon=horizon, parr=parr,
                           period_size=period_size)
    run_mean_sds[alg.name()].append(mean_regret)
  return run_mean_sds


"""Experiments"""
#@title 1 # Fig 1 (in Kveton 22)
# cascade model
def exper1(time_stamp, T, num_runs, L, k_list, reload_=False, T_plots=None, algs=None, envs=None,
    beta_a=None, beta_b=None, parr=0, period_size=1, expr_fname=None):

  env = PriorCMEnv(k_list, L, beta_a, beta_b)

  if not expr_fname:
    expr_fname = f"exp1-L{L}-K{k_list}-horizon{T}-nrun{num_runs}-{time_stamp}"

  pos_prior = {}

  def eval_1(alg):
    # import pdb; pdb.set_trace()
    fname = f"expr1-{alg.name()}-{PriorCMEnv.name()}-{time_stamp}"
    regret, _, pp_mean, pp_sd = evaluate(alg, env, num_runs, T,
                                         reload_=reload_, fname=fname,
                                         plot_pos=plot_pos, parr=parr
                                         , write_save=write_save, period_size=period_size,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)
    pos_prior[alg.name()] = {'mean': pp_mean, 'sd': pp_sd}
    cum_regret = np.cumsum(regret, axis=0)
    mean_vals = cum_regret.mean(axis=1)  # size: [horizon]
    se_vals = cum_regret.std(axis=1) / np.sqrt(num_runs)  # size: [horizon]
    return mean_vals, se_vals

  # Save all here.
  if PICKLE_LOAD:
    mean_sds = load_res(expr_fname)
  else:
    mean_sds = {}
    for alg in algs:
      mean_sds[alg.name()] = eval_1(alg)#, alg_colors[alg.name()], alg_line_styles[alg.name()])
    if PICKLE_SAVE:
      save_res(expr_fname, mean_sds)

  plot_exper(T_plots, algs, mean_sds, num_items=L, k_list=k_list, expr_fname=expr_fname,
             plot_pos=plot_pos, pos_prior=pos_prior, time_stamp=time_stamp, horizon=T,
             reload_=reload_, period_size=period_size)

#@title 2 # Fig 1
#%%
# Resample Beta prior
def exper2(time_stamp, T, num_runs, L, k_list, reload_=False,
    num_trials=5, runs_per_trial = 5, algs=None, envs=None, period_size=1):
  def eval_and_plot(regret, color, linestyle='-'):
    cum_regret = np.cumsum(regret, axis=1)
    mean_vals = cum_regret.mean(axis=0)  # size: [horizon]
    se_vals = cum_regret.std(axis=0) / np.sqrt(np.shape(regret)[0])  # size: [horizon]
    plt.fill_between(steps, mean_vals+se_vals, mean_vals-se_vals,
                     alpha = 0.1, edgecolor = "none", facecolor = color)
    plt.plot(mean_vals, color=color, label=alg.name(), linestyle=linestyle)

  # horizon = 2000
  # # horizon = 20
  # num_trials = 20
  # num_trials = 3
  # runs_per_trial = 20
  # runs_per_trial = 3
  # num_items = 30
  # k_list = 3
  # envs = [PriorCMEnv, PriorDCTREnv, DCMEnv]
  # envs = [PriorCMEnv]
  env_regrets = {}
  for env_type in envs:
    print("Envirnoment:", env_type.name())
    regrets = {}
    for alg in algs:
      regrets[alg.name()] = np.array([])
    beta_b = 10 * np.ones([L])
    for i in range(num_trials):
      print("Trial:", i, "out of", num_trials)
      beta_a = np.random.randint(low=1, high=10, size=[L])
      env = env_type(k_list, L, beta_a, beta_b)

      for alg in algs:
        fname = f"expr2-{alg.name()}-{env_type.name()}-t{i}-{time_stamp}"
        regret, _,_,_ = evaluate(alg, env, runs_per_trial, T, reload_=reload_,
                                 fname=fname, plot_pos=plot_pos, parr=parr,
                                 write_save=write_save, period_size=period_size,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)  # [horizon, runs_per_trial]
        regret = np.transpose(regret)  # [runs_per_trial, horizon]
        if len(regrets[alg.name()]):
          regrets[alg.name()] = np.concatenate((regrets[alg.name()], regret), axis=0)
        else:
          regrets[alg.name()] = regret
    env_regrets[env_type.name()] = regrets

    # regrets[alg] size: [num_trials * runs_per_trial, horizon]

  steps = range(T)
  for env_type in envs:
    plt.figure()
    regrets = env_regrets[env_type.name()]

    for alg in algs:
      eval_and_plot(regrets[alg.name()], alg_colors[alg.name()])


    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': 10})
    plt.xlabel("Round n", fontweight='bold', fontsize=12)
    plt.ylabel("Regret", fontweight='bold', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(env_type.name(), fontweight='bold', fontsize=14)
    # Save fig

    fig_file = "exp2-"+ env_type.name() + f'-L{L}-K{k_list}-horizon{T}-nrun{num_runs}-trial{num_trials}-rpt{runs_per_trial}-{time_stamp}.pdf'
    plt.tight_layout()
    # plt.savefig(fig_file, format = "pdf", bbox_inches = 0)
    save_fig(fig_file)
    # plt.show()
    plt.close()

  if 0:
    #%%
    # Save a copy
    import copy
    env_regrets_keep = copy.copy(env_regrets)


    #%%
    for env_type in envs:
      print("Env:", env_type.name())
      regrets = env_regrets[env_type.name()]
      print("toprank mean:", np.mean(np.sum(regrets[TopRank.name()], axis=1)))
      print("greedy mean:", np.mean(np.sum(regrets[Greedy.name()], axis=1)))


    #%%

    # alg_colors = {
    #     BayesUCB.name(): "orange",
    #     ThompsonSampling.name(): "green",
    #     Greedy.name(): "purple",
    #     CascadeUCB1.name(): "gray",
    #     CascadeKLUCB.name(): "blue",
    #     TopRank.name(): "red",
    #     fMeta_TS.name(): ""
    # }
    # alg_labels = {
    #     BayesUCB.name(): "BayesUCB",
    #     ThompsonSampling.name(): "TS",
    #     CascadeUCB1.name(): "CascadeUCB1",
    #     CascadeKLUCB.name(): "CascadeKL-UCB",
    #     TopRank.name(): "TopRank",
    #     Greedy.name(): "Greedy",
    #     fMeta_TS.name(): "fMeta_TS"
    # }


    #%% Plot only
    def eval_and_plot(regret, color, linestyle='-'):
      cum_regret = np.cumsum(regret, axis=1)
      mean_vals = cum_regret.mean(axis=0)  # size: [horizon]
      se_vals = cum_regret.std(axis=0) #/ np.sqrt(np.shape(regret)[0])  # size: [horizon]
      plt.fill_between(steps, mean_vals+se_vals, mean_vals-se_vals,
                       alpha = 0.1, edgecolor = "none", facecolor = color)
      plt.plot(mean_vals, color=color, label=alg_labels[alg.name()], linestyle=linestyle)

    for env_type in envs:
      plt.figure()
      regrets = env_regrets[env_type.name()]
      for alg in algs:
        eval_and_plot(regrets[alg.name()], alg_colors[alg.name()])

      plt.legend(loc='upper left', prop={'weight': 'bold', 'size': 10}, frameon=False)
      plt.xlabel("Round n", fontweight='bold', fontsize=12)
      plt.ylabel("Regret", fontweight='bold', fontsize=12)
      plt.locator_params(axis='x', nbins=5)
      plt.locator_params(axis='y', nbins=5)
      plt.xticks(fontsize=10)
      plt.yticks(fontsize=10)
      plt.title(env_type.name(), fontweight='bold', fontsize=14)
      plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
      # Save fig
      # FIGS_DIR = './tmp/'
      fig_file = FIGS_DIR + "exp2.2-"+ env_type.name() + f'-L{L}-K{k_list}-horizon{T}-nrun{num_runs}.pdf'
      plt.savefig(fig_file, format = "pdf", bbox_inches = 0)
      # plt.show()
      plt.close()


#@title 3 # Fig 2
#%%
#%% Bound vs Regret
def exper3(time_stamp, T, num_runs, L, k_list, reload_=False,
    ab_vals=None, algs=None, envs=None, period_size=1):

  algs = [BayesUCB, ThompsonSampling, GaussTS_LTR]

  # horizon = 500
  # horizon = 10
  # num_items = 30
  # k_list = 3
  b_factor = 10
  # num_runs = 500
  # num_runs = 5
  # ab_vals = np.array([1, 10, 20, 50, 100, 200, 500, 1000])
  # ab_vals = np.array([1, 10])

  expr_fname = f"exp3-reg_n_bnd-L{L}-K{k_list}-horizon{T}-bf{b_factor}-{time_stamp}"

  def get_final_regret(regret):
    cum_regret = np.cumsum(regret, axis=0)  # size: [horizon, num_runs]
    mean_cum_regret = np.mean(cum_regret, axis=1)
    stdev_cum_regret = np.std(cum_regret, axis=1) / np.sqrt(np.shape(regret)[1])
    return mean_cum_regret[-1], stdev_cum_regret[-1]

  # envs = [PriorCMEnv, PriorDCTREnv]
  # envs = [PriorCMEnv]

  # alg_colors = {
  #     BayesUCB.name(): "orange",
  #     ThompsonSampling.name(): "green",
  #     CascadeUCB1.name(): "gray",
  #     CascadeKLUCB.name(): "blue",
  #     TopRank.name(): "red",
  # }

  if PICKLE_LOAD:
    env_alg_ab_regrets = load_res(expr_fname)
  else:
    env_alg_ab_regrets = {}
    for env_type in envs:
      env_name = env_type.name()
      env_alg_ab_regrets[env_name] = {}
      for alg in algs:
        alg_name = alg.name()
        env_alg_ab_regrets[env_name][alg_name] = np.zeros((len(ab_vals)))

    for i, ab_val in enumerate(ab_vals):
      print("Gamma =", ab_val)
      beta_a = ab_val * np.ones([L])
      beta_b = b_factor * ab_val * np.ones([L])
      for env_type in envs:
        env = env_type(k_list, L, beta_a, beta_b)
        for bandit_alg in algs:
          fname = f"expr3-{bandit_alg.name()}-{env_type.name()}-t{i}-{time_stamp}"
          regret, _,_,_ = evaluate(bandit_alg, env, num_runs, T, reload_=reload_,
                                   fname=fname, plot_pos=plot_pos, parr=parr,
                                   write_save=write_save, period_size=period_size,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)  # [num_runs, horizon]
          last_regret, _ = get_final_regret(regret)
          env_alg_ab_regrets[env_type.name()][bandit_alg.name()][i] = last_regret
    save_res(expr_fname, env_alg_ab_regrets)

  bound1 = np.sqrt(k_list*L*T*np.log(T))/2 * np.sqrt(np.log(1 + T * 1.0/((b_factor+1)*ab_vals)))
  bound2 = np.sqrt(L) * np.log(T) * np.sqrt(T*k_list + L*(b_factor+1)*ab_vals) - L*np.log(T)*np.sqrt((b_factor+1)*ab_vals)

  # if reload_:
  #   env_alg_ab_regrets = load_res(fname=f"expr3-{time_stamp}")
  # else:
  #   save_res(fname=f"expr3-{time_stamp}", res=env_alg_ab_regrets)

  for env_type in envs:
    env_name = env_type.name()
    for alg in algs:
      alg_name = alg.name()
      label = env_name + ":" + alg_labels[alg_name]
      plt.semilogy(ab_vals, env_alg_ab_regrets[env_name][alg_name],
                   linewidth=2, linestyle=alg_line_styles[alg_name],
                   color=alg_colors[alg_name], label=label)
  plt.semilogy(ab_vals, bound1, linewidth=2, color='red', label='Bound')
  #plt.semilogy(ab_vals, bound2, linewidth=2, color='blue', label='Bound2')

  plt.xlabel("Gamma", fontweight='bold', fontsize=12)
  plt.ylabel("Regret", fontweight='bold', fontsize=12)
  plt.legend(loc='upper right', prop={'weight': 'bold', 'size': 10})
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.title("Bound and Actual Regret", fontweight='bold', fontsize=14)
  plt.gcf().subplots_adjust(bottom=0.15)
  # Save fig
  # fig_file = f"./tmp/exp3-compare_regret_n_bnd-num_items{num_items}-K{k_list}-horizon{horizon}-bf{b_factor}-ab{'-'.join([str(xx) for xx in ab_vals])}-{time_stamp}.pdf"
  fig_file = expr_fname + ".pdf"
  # plt.savefig(fig_file, format="pdf", bbox_inches=0)
  save_fig(fig_file)
  # plt.show()
  plt.close()

  if 0:
    #%% Plot only
    bound1 = np.sqrt(k_list*L*T*np.log(T))/2 * np.sqrt(np.log(1 + T * 1.0/((b_factor+1)*ab_vals)))
    bound2 = np.sqrt(L) * np.log(T) * np.sqrt(T*k_list + L*(b_factor+1)*ab_vals) - L*np.log(T)*np.sqrt((b_factor+1)*ab_vals)

    for env_type in envs:
      env_name = env_type.name()
      for alg in algs:
        alg_name = alg.name()
        label = env_name + ":" + alg_name
        plt.semilogy(ab_vals, env_alg_ab_regrets[env_name][alg_name],
                     linewidth=2, linestyle=env_styles[env_name],
                     color=alg_colors[alg_name], label=label)
    plt.semilogy(ab_vals, bound1, linewidth=2, color='red', label='Bound')
    #plt.semilogy(ab_vals, bound2, linewidth=2, color='blue', label='Bound2')

    plt.xlabel("Gamma", fontweight='bold', fontsize=12)
    plt.ylabel("Regret", fontweight='bold', fontsize=12)
    plt.legend(loc='upper right', prop={'weight': 'bold', 'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Bound and Actual Regret", fontweight='bold', fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.15)
    # Save fig
    fig_file = './tmp/exp3-compare_regret_and_bound2.pdf'
    plt.savefig(fig_file, format="pdf", bbox_inches=0)
    # plt.show()


#@title 4 Fig 2 ish
#%% Bound vs Regret: high attractions
def exper4(time_stamp, T, num_runs, L, k_list, reload_=False,
    algs=None, envs=None, period_size=1):
  """compare_regret_and_bound_high_attractions"""

  # envs = [PriorCMEnv, PriorDCTREnv]
  # algs = [BayesUCB, ThompsonSampling, fMeta_TS, GaussTS]

  # horizon = 500
  # num_items = 30
  # k_list = 3
  gammas = np.arange(0, 10)
  # num_runs = 500

  def get_final_regret(regret):
    cum_regret = np.cumsum(regret, axis=0)  # size: [horizon, num_runs]
    mean_cum_regret = np.mean(cum_regret, axis=1)
    stdev_cum_regret = np.std(cum_regret, axis=1) / np.sqrt(np.shape(regret)[1])
    return mean_cum_regret[-1], stdev_cum_regret[-1]


  # alg_colors = {
  #     BayesUCB.name(): "orange",
  #     ThompsonSampling.name(): "green",
  #     CascadeUCB1.name(): "gray",
  #     CascadeKLUCB.name(): "blue",
  #     TopRank.name(): "red",
  # }

  env_alg_ab_regrets = {}
  for env_type in envs:
    env_name = env_type.name()
    env_alg_ab_regrets[env_name] = {}
    for alg in algs:
      alg_name = alg.name()
      env_alg_ab_regrets[env_name][alg_name] = np.zeros((len(gammas)))

  for i, gamma in enumerate(gammas):
    print("Gamma =", gamma)
    beta_a = (1+gamma) * np.ones([L])
    beta_b = (10-gamma) * np.ones([L])
    for env_type in envs:
      env = env_type(k_list, L, beta_a, beta_b)
      for bandit_alg in algs:
        alg_name = bandit_alg.name()
        fname = f"expr4-{alg_name}-{env_type.name()}-t{i}-{time_stamp}"
        regret, _,_,_ = evaluate(bandit_alg, env, num_runs, T,
                                 reload_=reload_, fname=fname,
                                 plot_pos=plot_pos, parr=parr,
                                 write_save=write_save, period_size=period_size,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)  # [num_runs, horizon]
        last_regret, _ = get_final_regret(regret)
        env_alg_ab_regrets[env_type.name()][alg_name][i] = last_regret

  bound = np.sqrt(k_list*L*T*np.log(T))/2 * np.sqrt(np.log(1 + T * 1.0/(1+gammas+10-gammas)))
  #bound2 = np.sqrt(num_items) * np.log(horizon) * np.sqrt(horizon*k_list + num_items*(b_factor+1)*ab_vals) - num_items*np.log(horizon)*np.sqrt((b_factor+1)*ab_vals)


  for env_type in envs:
    env_name = env_type.name()
    for alg in algs:
      alg_name = alg.name()
      label = env_name + ":" + alg_name
      plt.semilogy(gammas, env_alg_ab_regrets[env_name][alg_name],
                   linewidth=2, linestyle=env_styles[env_name],
                   color=alg_colors[alg_name], label=label)
  plt.semilogy(gammas, bound, linewidth=2, color='red', label='Bound')
  #plt.semilogy(ab_vals, bound2, linewidth=2, color='blue', label='Bound2')

  plt.xlabel("Gamma", fontweight='bold', fontsize=12)
  plt.ylabel("Regret", fontweight='bold', fontsize=12)
  plt.legend(loc='lower left', prop={'weight': 'bold', 'size': 10})
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.title("Bound and Actual Regret", fontweight='bold', fontsize=14)
  plt.gcf().subplots_adjust(bottom=0.15,left=0.15)
  # Save fig
  # fig_file = f'./tmp/exp4-compare_regret_and_bound_high_attractions1-{time_stamp}.pdf'
  fig_file = f'exp4-L{L}-K{k_list}-horizon{T}-nrun{num_runs}-{time_stamp}.pdf'
  # plt.savefig(fig_file, format="pdf", bbox_inches = 0)
  save_fig(fig_file)
  # plt.show()
  plt.close()

  if 0:
    #%% Plot only
    for env_type in envs:
      env_name = env_type.name()
      for alg in algs:
        alg_name = alg.name()
        label = env_name + ":" + alg_name
        plt.semilogy(gammas, env_alg_ab_regrets[env_name][alg_name],
                     linewidth=2, linestyle=env_styles[env_name],
                     color=alg_colors[alg_name], label=label)
    plt.semilogy(gammas, bound, linewidth=2, color='red', label='Bound')
    #plt.semilogy(ab_vals, bound2, linewidth=2, color='blue', label='Bound2')

    plt.xlabel("Gamma", fontweight='bold', fontsize=12)
    plt.ylabel("Regret", fontweight='bold', fontsize=12)
    plt.legend(loc='lower left', prop={'weight': 'bold', 'size': 10}, frameon=False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Bound and Actual Regret", fontweight='bold', fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.15,left=0.15)
    # Save fig
    fig_file = 'exp4-compare_regret_and_bound_high_attractions2.pdf'
    plt.savefig(fig_file, format = "pdf", bbox_inches = 0)
    # plt.show()


#@title 5 # Fig 2 ish
#%% Misspecified Prior
# cascade model
def exper5(time_stamp, T, num_runs, L, k_list, reload_=False, period_size=1):
  """misspec_regret"""

  expr_fname = f'exp5-misspec_regret-L{L}-K{k_list}-horizon{T}-nrun{num_runs}-{time_stamp}'

  # algs = [BayesUCB, ThompsonSampling, fMeta_TS, GaussTS]
  algs = [BayesUCB, ThompsonSampling,
          # GaussTS_LTR,
          GPTSmean
          ]
  # misspecifications = range(0, 10)
  misspecifications = range(0, 10, 2)
  # misspecifications = range(0, 4, 2)

  beta_a = 1 * np.ones([L])
  beta_b = 10 * np.ones([L])
  env = PriorCMEnv(k_list, L, beta_a, beta_b)

  steps = range(T)

  def get_final_regret(regret):
    cum_regret = np.cumsum(regret, axis=0)  # size: [horizon, num_runs]
    mean_cum_regret = np.mean(cum_regret, axis=1)
    stdev_cum_regret = np.std(cum_regret, axis=1) / np.sqrt(np.shape(regret)[1])
    return mean_cum_regret[-1], stdev_cum_regret[-1]


  if PICKLE_LOAD:
    alg_misspec_regrets = load_res(expr_fname)
  else:
    alg_misspec_regrets = {}
    for alg in algs:
      alg_name = alg.name()
      alg_misspec_regrets[alg_name] = [None]*len(misspecifications)
      for misc_idx, misspec in enumerate(misspecifications):
        print("Misspecification:", misspec)
        fname = f"expr5-{alg.name()}-{PriorCMEnv.name()}-mis{misspec}-{time_stamp}"
        regret, _, _, _ = evaluate(alg, env, num_runs, T,
                                 misspecification=misspec,
                                 reload_=reload_, fname=fname, period_size=period_size,
                                 plot_pos=plot_pos, parr=parr, write_save=write_save,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)  # [num_runs, horizon]
        last_regret, last_regret_sd = get_final_regret(regret)
        alg_misspec_regrets[alg_name][misc_idx] = (last_regret, last_regret_sd)
    if PICKLE_SAVE:
      save_res(expr_fname, alg_misspec_regrets)

  #%%
  # Plot
  alg_line_styles[BayesUCB.name()]= 'dashdot'
  alg_line_styles[GPTSmean.name()]= '-'

  for alg in algs:
    alg_name = alg.name()
    me_sd = [None, None]
    me_sd[0] = [alg_misspec_regrets[alg_name][misc_idx][0] for misc_idx in range(len(misspecifications))]
    me_sd[1] = [alg_misspec_regrets[alg_name][misc_idx][1] for misc_idx in range(len(misspecifications))]
    me_sd = [np.asarray(me_sd[0]), np.asarray(me_sd[1])]
    plt.plot(misspecifications, me_sd[0],
             linewidth=2, color=alg_colors[alg_name], label=alg_labels[alg_name],
             linestyle=alg_line_styles[alg_name])
    plt.fill_between(misspecifications,
                     me_sd[0] + 3*me_sd[1],
                     me_sd[0] - 3*me_sd[1],
                     alpha=0.1, edgecolor="none", facecolor=alg_colors[alg_name])

  plt.xlabel('Misspecification', fontweight='bold', fontsize=12)
  plt.ylabel("Regret", fontweight='bold', fontsize=12)
  plt.legend(loc='upper left', prop={'weight': 'bold', 'size': 10}, frameon=False)
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  # plt.title("Regret vs Prior Misspecification", fontweight='bold', fontsize=14)
  plt.gcf().subplots_adjust(bottom=0.2)
  # Save fig
  fig_file = expr_fname + '.pdf'
  # plt.savefig(fig_file, format = "pdf", bbox_inches = 0)
  save_fig(fig_file)
  # plt.show()
  plt.close()


#@title 6 # Fig 3
#%%
# cascade model
def exper6(time_stamp, T, num_runs, L, k_list, reload_=False,
    algs=None, envs=None,  period_size=1):
  # horizon = 3000
  # num_runs = 50
  # num_items = 30
  # k_list = 3
  beta_a = 1 * np.ones([L])
  beta_b = 10 * np.ones([L])

  env = PriorCMEnv(k_list, L, beta_a, beta_b)

  expr_fname = f'exp6-misspec_cum_reg_baseline-L{L}-K{k_list}-horizon{T}-nrun{num_runs}-{time_stamp}'

  steps = range(T)

  def eval_and_plot(alg, color=None, linestyle='-', misspecification=0, label=None):
    if not color:
      color = alg_colors[alg.name()]
    fname = f"expr6-{alg.name()}-{PriorCMEnv.name()}-mis{misspecification}-lb{label}-{time_stamp}"

    if PICKLE_LOAD:
      regret = load_res(fname)
    else:
      regret, _,_,_ = evaluate(alg, env, num_runs, T,
                               misspecification=misspecification,
                               reload_=reload_, fname=fname,
                               plot_pos=plot_pos, parr=parr,
                               write_save=write_save, period_size=period_size,
             load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res)
      if PICKLE_SAVE:
        save_res(fname, regret)

    cum_regret = np.cumsum(regret, axis=0)
    mean_vals = cum_regret.mean(axis=1)  # size: [horizon]
    se_vals = cum_regret.std(axis=1) / np.sqrt(num_runs)  # size: [horizon]
    plt.fill_between(steps, mean_vals+se_vals, mean_vals-se_vals,
                     alpha = 0.1, edgecolor = "none", facecolor = color)
    if not label:
      label = "c=" + str(misspecification)
    plt.plot(mean_vals, color=color, label=label, linestyle=linestyle,
             linewidth=2)



  # for alg in algs:
  #   eval_and_plot(alg, label=alg_labels[alg.name()])

  eval_and_plot(BayesUCB, "orange", misspecification=0, label=alg_labels[BayesUCB.name()])
  eval_and_plot(BayesUCB, "orange", misspecification=4, label=alg_labels[BayesUCB.name()])
  eval_and_plot(BayesUCB, "orange", misspecification=9, label=alg_labels[BayesUCB.name()])

  # eval_and_plot(CascadeKLUCB, label=alg_labels[CascadeKLUCB.name()])
  # eval_and_plot(CascadeUCB1, label=alg_labels[CascadeUCB1.name()])

  eval_and_plot(ThompsonSampling, linestyle='-', misspecification=0, label=alg_labels[ThompsonSampling.name()])
  eval_and_plot(ThompsonSampling, linestyle='--', misspecification=4, label=alg_labels[ThompsonSampling.name()])
  eval_and_plot(ThompsonSampling, linestyle=':', misspecification=9, label=alg_labels[ThompsonSampling.name()])
  # eval_and_plot(fMeta_TS, linestyle='-', misspecification=0)
  # eval_and_plot(fMeta_TS, linestyle='--', misspecification=4)
  # eval_and_plot(fMeta_TS, linestyle=':', misspecification=9)
  eval_and_plot(GaussTS_LTR, linestyle='-', misspecification=0, label=alg_labels[GaussTS_LTR.name()])
  eval_and_plot(GaussTS_LTR, linestyle='--', misspecification=4, label=alg_labels[GaussTS_LTR.name()])
  eval_and_plot(GaussTS_LTR, linestyle=':', misspecification=9, label=alg_labels[GaussTS_LTR.name()])

  # eval_and_plot(GPTS, linestyle='-', misspecification=0, label=alg_labels[GPTS.name()])
  # eval_and_plot(GPTS, linestyle='--', misspecification=4, label=alg_labels[GPTS.name()])
  # eval_and_plot(GPTS, linestyle=':', misspecification=9, label=alg_labels[GPTS.name()])

  # eval_and_plot(GPTSmean, linestyle='-', misspecification=0, label=alg_labels[GPTSmean.name()])
  # eval_and_plot(GPTSmean, linestyle='--', misspecification=4, label=alg_labels[GPTSmean.name()])
  # eval_and_plot(GPTSmean, linestyle=':', misspecification=9, label=alg_labels[GPTSmean.name()])

  plt.legend(loc='upper left', prop={'weight': 'bold', 'size': 10}, frameon=False)
  plt.xlabel("Round n", fontweight='bold', fontsize=12)
  plt.ylabel("Regret", fontweight='bold', fontsize=12)
  plt.title('Prior Misspecification', fontweight='bold', fontsize=14)
  plt.locator_params(axis='x', nbins=8)
  plt.locator_params(axis='y', nbins=8)
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
  # Save fig
  fig_file = expr_fname + '.pdf'
  # plt.savefig(fig_file, format="pdf", bbox_inches=0)
  save_fig(fig_file)
  # plt.show()
  plt.close()

#@title 7 # Prior for GTS

# cascade model
def exper7(time_stamp, T, num_runs, L, K, reload=False, T_plots=None,
    period_size=1, expr_fname=None):

  beta_a = np.random.randint(low=1, high=10, size=[L])
  beta_b = 10 * np.ones([L])
  env = PriorCMEnv(K, L, beta_a, beta_b)

  tmp_algs = [GaussTS_LTR]
  GTS_priors = [#[0,1],
                [0, 1/4],
      #[.5, 1], [.5, 1/4], ['BetaMean', 1],
      ['BetaMean', 1/4], ['BetaMean', 'BetaVar']
  ]  # be careful when reloading if you change this

  tmp = ['-', '--', ':', '-.', (0,(1,3)), (0, (5, 2)), (0,(1,3,5,5)), 'dashdot', 'dotted']
  GTS_prior_style = {str(k): tmp[kcnt] for kcnt, k in enumerate(GTS_priors)}
  tmp = ['r', 'g', 'b', 'orange', 'k', 'm', 'y']
  GTS_prior_clr = {str(k): tmp[kcnt] for kcnt, k in enumerate(GTS_priors)}
  if not expr_fname:
    expr_fname = f"exp7-num_items{L}-K{K}-horizon{T}-nrun{num_runs}-{time_stamp}"

  pos_prior = {}

  def eval_1(alg, GTS_prior):
    fname = f"expr7-{alg.name()}-{PriorCMEnv.name()}-{time_stamp}-{GTS_prior}"
    regret, _, pp_mean, pp_sd = evaluate(alg, env, num_runs, T,  period_size=period_size,
                                         reload_=reload_, fname=fname, GTS_prior=GTS_prior)
    pos_prior[alg.name()] = {'mean': pp_mean, 'sd': pp_sd}
    cum_regret = np.cumsum(regret, axis=0)
    mean_vals = cum_regret.mean(axis=1)  # size: [horizon]
    se_vals = cum_regret.std(axis=1) / np.sqrt(num_runs)  # size: [horizon]
    return mean_vals, se_vals


  # Save all here.
  if PICKLE_LOAD:
    mean_sds = load_res(expr_fname)
  else:
    mean_sds = {}
    for alg in tmp_algs:
      mean_sds[alg.name()] = {}
      # for cntr, GTS_prior in enumerate(GTS_priors):
      for GTS_prior in GTS_priors:
        mean_sds[alg.name()][str(GTS_prior)] = eval_1(alg, GTS_prior)
        print(GTS_prior)
    if PICKLE_SAVE:
      save_res(expr_fname, mean_sds)

  for tt in T_plots:
    for alg, me_sdalg in mean_sds.items():
      # for GTS_prior_cntr, me_sd in me_sdalg.items():
      for GTS_prior, me_sd in me_sdalg.items():
        plt.fill_between(range(tt),
                         me_sd[0][:tt] + me_sd[1][:tt],
                         me_sd[0][:tt] - me_sd[1][:tt],
                         alpha=0.1, edgecolor="none", facecolor=alg_colors[alg])
        plt.plot(me_sd[0][:tt], color=GTS_prior_clr[GTS_prior],
                 # label=str(GTS_priors[GTS_prior]),
                 label=GTS_prior,
                 linestyle=GTS_prior_style[GTS_prior], linewidth=3)
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':10}, markerscale=5)
    plt.xlabel("Round", fontweight='bold', fontsize=10)
    plt.ylabel("Cumulative Regret", fontweight='bold', fontsize=10)
    # plt.title("Prior CM, GTS prior, num_items={}, K={}".format(L, K))
    plt.tight_layout()
    fig_file = expr_fname + f"-tt{tt}.pdf"
    save_fig(fig_file)
    plt.close()


def init_env(k_list, num_items, dim, ex_type, random_seed, env_type, reset=False):
  """Initialize the environment."""
  if reset:
    np.random.seed(110)
  else:
    if random_seed:
      np.random.seed(random_seed)  # IMPORTANT for parr=1

  sigma_0 = 0.1 * np.ones(dim)
  sigma = 1/4
  if ex_type == linear_ex_type:
    theta = np.random.rand(dim)
    theta /= np.linalg.norm(theta)
  else:
    mu_theta = np.random.randn(dim)
    mu_theta = np.zeros(dim)
    # sample problem instance
    theta = mu_theta + sigma_0 * np.random.randn(dim)
    # theta = np.random.rand(dim)
    theta /= np.linalg.norm(theta)

  # sample arms from a unit ball
  # X = np.random.randn(num_items, dim)
  X = np.random.rand(num_items, dim)
  X /= np.linalg.norm(X, axis=-1)[:, np.newaxis]

  beta_a = np.random.randint(low=1, high=10, size=[num_items])
  # beta_a = 10 * np.ones([num_items])
  beta_b = 10 * np.ones([num_items])

  if ex_type == linear_ex_type and env_type == LTREnv_features.name():
    click_probs = X.dot(theta)
    assert (np.where(0 > click_probs)[0].shape[0] +
            np.where(1 < click_probs)[0].shape[0] == 0), (
        "linear click probs wrong!")
  elif ex_type == log_ex_type:# and env_type == LogBandit_LTR_Env.name():
    click_probs = 1 / (1 + np.exp(- X.dot(theta)))
    beta_b = 10 * np.ones([num_items])
    beta_a = 10 * click_probs/(1-click_probs)
  else:
    click_probs = np.random.beta(beta_a, beta_b)

  # print(click_probs)
  if ex_type == stand_ex_type:
    if env_type in [PriorCMEnv.name(), PriorDCTREnv.name(), DCMEnv.name()]:
      # env = PriorCMEnv(k_list=k_list, num_items=num_items, beta_a=beta_a, beta_b=beta_b)
      env = eval(f"{env_type}Env")
      env = env(k_list=k_list, L=num_items, beta_a=beta_a, beta_b=beta_b)
    else:
      raise NotImplementedError(f"{env_type} not implemented.")
  elif ex_type == linear_ex_type:
    if env_type == LinBandit_LTR_Env.name():
      env = LinBandit_LTR_Env(k_list=k_list,
                              L=num_items,
                              # preds=[],
                              X=X,
                              theta=theta, sigma=sigma, beta_a=beta_a,
                              beta_b=beta_b)
      # env_type = LinBandit_LTR_Env
    elif env_type == LTREnv_features.name():
      env = LTREnv_features(X=X, beta_a=beta_a, beta_b=beta_b,
                            click_probs=click_probs,
                            scores_sum=[], scores_sq=[],
                            k_list=k_list)
      # env_type = LTREnv_features
    else:
      raise NotImplementedError(f"{env_type} not implemented.")
  elif ex_type == log_ex_type:
    if env_type == LogBandit_LTR_Env.name():
      env = LogBandit_LTR_Env(k_list=k_list, X=X, theta=theta,
                              beta_a=beta_a, beta_b=beta_b)
      # env_type = LogBandit_LTR_Env
    elif env_type == LTREnv_features.name():
      env = LTREnv_features(X=X, beta_a=beta_a, beta_b=beta_b,
                            click_probs=click_probs, scores_sum=[], scores_sq=[],
                            k_list=k_list)
      # env_type = LTREnv_features
    else:
      raise NotImplementedError(f"{env_type} not implemented.")
  else:
    raise NotImplementedError(f"{ex_type} not implemented.")

  return env #, env_type


def exper_cntx_ltr(time_stamp, horizon, num_runs, num_items, k_list, d=5,
    reload_=False, T_plots=None, algs=None, parr=0,
    period_size=1, ex_type=None, env_type=None, expr_fname=None):
  # exper-8

  print("exper_cntx_ltr ", ex_type)
  exp_name = "exp-cntx"

  env = init_env(k_list=k_list, num_items=num_items, dim=d, ex_type=ex_type,
                 random_seed=None, env_type=env_type, reset=False)

  if not expr_fname:
    expr_fname = f"{exp_name}-{env_type}-L{num_items}-K{k_list}" \
               f"-d{d}-horizon{horizon}-nrun{num_runs}-{time_stamp}"

  pos_prior = {}

  def eval_1(alg):
    fname = f"expr-cntx-ltr-{alg.name()}-{time_stamp}"
    regret, _, pp_mean, pp_sd = evaluate(
        alg, env, num_runs, horizon, reload_=reload_, fname=fname,
        plot_pos=plot_pos, parr=parr, write_save=write_save, period_size=period_size,
        load_res=load_res, num_cpu=num_cpu, mp=mp, save_res=save_res, _verbose=True)
    pos_prior[alg.name()] = {'mean': pp_mean, 'sd': pp_sd}
    cum_regret = np.cumsum(regret, axis=0)
    mean_vals = cum_regret.mean(axis=1)  # size: [horizon]
    se_vals = cum_regret.std(axis=1) / np.sqrt(num_runs)  # size: [horizon]
    return mean_vals, se_vals

  # Save all here.
  if PICKLE_LOAD:
    mean_sds = load_res(expr_fname)
  else:
    mean_sds = {}
    for alg in algs:
      print(alg)
      mean_sds[alg.name()] = eval_1(alg)
      # eval_alg_env(alg, env, time_stamp, num_runs, exp_name, horizon, parr,
      #              num_cpu, mp, period_size=1)
    if PICKLE_SAVE:
      save_res(expr_fname, mean_sds)

  plot_exper(T_plots, algs, mean_sds, num_items=num_items, k_list=k_list,
             expr_fname=expr_fname, period_size=period_size,
             plot_pos=plot_pos, pos_prior=pos_prior, time_stamp=time_stamp,
             horizon=horizon, reload_=False)


def exper_cntx_ltr_Bayes(time_stamp, horizon, num_runs, num_items, k_list, dim=5,
    reload_=False, T_plots=None, algs=None, period_size=1, ex_type=None,
    num_cpu=None, parr=0, env_type=None, reset=False, expr_fname=None):
  # exper-9
  """Regenerate the environment in each run (Bayes Regret)"""

  print("exper_cntx_ltr_Bayes ", ex_type)

  exp_name = "cntx_Bayes"
  if expr_fname:
    expr_fname = expr_fname
  else:
    expr_fname = f"{exp_name}-{ex_type}-{env_type}-num_items{num_items}-K{k_list}" \
               f"-d{dim}-horizon{horizon}-nrun{num_runs}-{time_stamp}"

  # Save all here.
  if PICKLE_LOAD:
    mean_sds = load_res(expr_fname)
  else:
    mean_sds = defaultdict(list)
    seeds = np.random.randint(2 ** 15 - 1, size=num_runs)
    if parr:
      print(f"MP running {num_cpu}!")
      def collect_result(run_mean_sds):
        for alg, regret in run_mean_sds.items():
          mean_sds[alg].append(regret)
      pool = mp.Pool(num_cpu)
      poolobjs = [pool.apply_async(eval_algs_env_bayes,
                                   args=[run_num, algs, ex_type, k_list, num_items, dim, horizon, period_size,
                seeds[run_num], env_type, reset, parr], callback=collect_result) for run_num
                  in range(num_runs)]
      print("before close"); pool.close(); pool.join()
      # for f in poolobjs:
      #   print(f.get())  # print the errors
    else:
      for run_num in range(num_runs):
        print("run_num ", run_num)  # , 10*"===")
        run_mean_sds = eval_algs_env_bayes(run_num, algs, ex_type, k_list, num_items, dim,
                                           horizon, period_size, seeds[run_num],
                                           env_type, reset, parr)
        for alg in algs:
          mean_sds[alg.name()].extend(run_mean_sds[alg.name()])

    for alg in algs:
      tmp = np.asarray(mean_sds[alg.name()]).reshape((horizon, num_runs))
      mean_sds[alg.name()] = regret_to_cumregret(tmp, num_runs)

    if PICKLE_SAVE:
      save_res(expr_fname, mean_sds)

  plot_exper(T_plots, algs, mean_sds, num_items, k_list, expr_fname,
             period_size=period_size)

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string("ex_type", "stand_ex_type", "Experiment type.")
flags.DEFINE_integer("expr_num", 8, "Experiment number/id.")


#@title Run
def main(argv):
  time_stamp = get_pst_time()
  make_dir(FIGS_DIR, PROJ_SAVE_DIR)
  make_dir(RES_DIR, PROJ_SAVE_DIR)

  algs = [GaussTS_LTR, BayesUCB, BayesUCBPriorOnly,
          ThompsonSampling, ThompsonSamplingPriorOnly,
          CascadeUCB1, CascadeKLUCB, TopRank, LinTS_LTR,
          LinTS_Cascade, LogTS_LTR, Cascade_LinUCB, CascadeWOFUL, LogTS_LTR]

  expr_num = -1  # the experiment number (id)
  # expr_num = 5  # the prior misc experiment

  ex_type = None  # experiment type
  ex_type = stand_ex_type; expr_num = 1
  ex_type = linear_ex_type; expr_num = 8
  ex_type = log_ex_type; expr_num = 8

  ex_type = eval(FLAGS.ex_type); expr_num = FLAGS.expr_num

  freq_flg = False  # if true, then reset the env in each run (frequentist)
  freq_flg = True

  if expr_num in [-1, 7, 5]:
    env_type = PriorCMEnv.name()
    env_type = None
  elif ex_type == stand_ex_type and expr_num == 1:
    expr_num = 1
    env_type = PriorCMEnv.name()
    # env_type = PriorDCTREnv.name()
    # env_type = DCMEnv.name()
    algs = [
        # LogTS_LTR,
        # Greedy,
        # Ensemble
        GaussTS_LTR,
        # TS_Cascade,
        BayesUCB, ThompsonSampling,
        GPTS,
        GPTSmean,
        # LinTS_LTR, LinTS_Cascade,
        # Cascade_LinUCB, CascadeWOFUL,
    ]
  elif ex_type == linear_ex_type and expr_num == 8:
    expr_num = 8
    env_type = LTREnv.name()
    env_type = LTREnv_features.name()
    # env_type = LinBandit_LTR_Env.name()
    algs = [
        LinTS_LTR,
        GaussTS_LTR,
        TS_Cascade,
        # BayesUCB, ThompsonSampling, # they need click
        # GPTS, GPTSmean,
        LinTS_Cascade,
        # Cascade_LinUCB,
        # CascadeWOFUL,
        # Greedy,
        # Ensemble
    ]
  elif ex_type == log_ex_type and expr_num == 8:
    expr_num = 8
    env_type = LTREnv_features.name()
    env_type = LogBandit_LTR_Env.name()
    algs = [
        LogTS_LTR,
        LinTS_LTR,
        # # Greedy,
        # # Ensemble
        GaussTS_LTR,
        TS_Cascade,
        BayesUCB, ThompsonSampling,
        # # GPTS, GPTSmean,
        # # LinTS_Cascade,
        Cascade_LinUCB,
        CascadeWOFUL,
    ]
  else:
    raise NotImplementedError(f"{ex_type} not implemented!")


  # envs = [PriorCMEnv, PriorDCTREnv, DCMEnv]
  envs = [PriorCMEnv]

  global PICKLE_SAVE, PICKLE_LOAD, \
    parr, plot_pos, write_save, reload_

  parr = 1
  parr = 0  # does not work well in Colab.
  reload_ = False; write_save = 0  # write_save = 1
  plot_pos = 0  # plot_pos = 1  # diff; # plot_pos = 2  # ratio diff
  PICKLE_SAVE = True
  # PICKLE_SAVE = False
  PICKLE_LOAD = False
  # PICKLE_LOAD = True  # set the time step and exp_fname below, also algs above

  reload_ = True; reload_ = False  # reload old npy type (DEPREC)
  exp_fname = None
  if reload_ or PICKLE_LOAD:
    """set the time_stamp, and if needed exp_fname (without -tt#.pkl), L, K, etc."""
    # exp_fname = "cntx_Bayes-Log-exp-L100-K10-d10-horizon10000-nrun100-07_13_2022_00_49_11"
    # time_stamp = "06-07-2022-22-35-31"
    # time_stamp = "07_14_2022_22_32_09"
    # exp_fname = "exp-cntx-LogBandit_LTR_Env-L20-K3-d10-horizon10000-nrun100-07_14_2022_22_32_09"
    # time_stamp = "07_14_2022_04_11_03"
    # exp_fname = "exp1-L100-K10-horizon10000-nrun100-07_14_2022_04_11_03"
    # time_stamp = "07_15_2022_17_47_45"
    # exp_fname = "exp1-L20-K3-horizon10000-nrun100-07_15_2022_17_47_45"
    # time_stamp = "06-10-2022-19-12-54"
    # exp_fname = "exp7-L20-K3-T10000-nrun100-06-10-2022-19-12-54"
    # time_stamp = "07_15_2022_04_18_33"
    # time_stamp = "07_20_2022_00_33_22"
    # time_stamp = "07_19_2022_23_10_29"
    # time_stamp = "07_20_2022_00_47_38"
    # time_stamp = "07_15_2022_17_47_45"
    # time_stamp = "07_15_2022_02_20_18"
    # time_stamp = "07_20_2022_03_08_57"
    # time_stamp = "07_20_2022_04_31_35"
    # time_stamp = "07_20_2022_05_31_48"
    # time_stamp = "07_22_2022_23_49_53"
    # time_stamp = "07_23_2022_00_46_33"
    # time_stamp = "07_14_2022_04_11_03"
    # time_stamp = "07_20_2022_00_47_38"
    # time_stamp = "07_25_2022_17_23_13"
    # time_stamp = "07_25_2022_17_52_35"
    time_stamp = "07_26_2022_00_12_10"

  test_run = 1
  # test_run = 0

  if test_run:
    period_size = 1
    horizon = 10000; T_plots = [horizon//10, horizon]
    num_items = 30; k_list = 3; dim = 5
    num_runs = 100
    num_trials = 5; runs_per_trial = 1
  else:
    period_size = 1
    horizon = 10000; T_plots = [horizon//10, horizon]
    # horizon = 10000; T_plots = [horizon]
    num_items = 30; k_list = 3; dim = 5
    # num_items = 100; k_list = 10; dim = 10
    num_runs = 100; num_trials = 20; runs_per_trial = 20

  assert k_list <= num_items, "k_list > num_items"
  # T_plots = [x//period_size for x in T_plots]
  assert period_size == 1, "period size > 1 not implemented"

  # expr3
  ab_vals = np.array([1, 10, 20, 50, 100, 200, 500, 1000])
  ab_vals = np.array([1, 10, 50, 100, 500, 1000])
  # ab_vals = np.array([1, 10])
  # Beta s
  beta_a = np.random.randint(low=1, high=10, size=[num_items])
  beta_b = 10 * np.ones([num_items])
  # print(beta_a[37], -np.sort(-beta_a))

  print("time_stamp", time_stamp)

  """Experiments."""
  if expr_num == 1:
    exper1(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_,
           T_plots=T_plots, algs=algs,
           envs=envs, beta_a=beta_a, beta_b=beta_b, parr=parr,
           period_size=period_size, expr_fname=exp_fname)  # Fig 1 in Kveton 22'
  elif expr_num == 2:
    exper2(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_, num_trials=num_trials,
           runs_per_trial = runs_per_trial, algs=algs, envs=envs,
           period_size=period_size)  # Fig 1
  elif expr_num == 3:
    exper3(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_, ab_vals=ab_vals,
           algs=algs, envs=envs, period_size=period_size) # Fig 2
  elif expr_num == 4:
    exper4(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_,
           algs=algs, envs=envs, period_size=period_size) # Fig 2 ish not really helpful
  elif expr_num == 5:
    exper5(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_,
           period_size=period_size)  # Fig 2 ish
  elif expr_num == 6:
    exper6(time_stamp, horizon, num_runs, num_items, k_list, reload_=reload_,
           algs=algs, envs=envs, period_size=period_size) # Fig 3
  elif expr_num == 7:
    exper7(time_stamp, horizon, num_runs, num_items, k_list, T_plots=T_plots,
           period_size=period_size, expr_fname=exp_fname)  # Prior for GTS
  elif expr_num == 8:
    exper_cntx_ltr(time_stamp, horizon, num_runs, num_items, k_list, dim, reload_, T_plots=T_plots,
                   algs=algs, parr=parr, period_size=period_size, ex_type=ex_type, env_type=env_type,
                   expr_fname=exp_fname)
  elif expr_num == 9:
    exper_cntx_ltr_Bayes(time_stamp, horizon, num_runs, num_items, k_list, dim, reload_,
                         T_plots=T_plots, algs=algs, period_size=period_size,
                         ex_type=ex_type, num_cpu=num_cpu, parr=parr, env_type=env_type,
                         reset=freq_flg, expr_fname=exp_fname)

if __name__ == "__main__":
  app.run(main)
