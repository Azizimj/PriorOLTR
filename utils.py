import numpy as np

def save_res(fname, regs, sregs):
    np.savetxt(fname, regs)
    np.savetxt(fname, sregs)

def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (1, 2)
  elif style == "-.":
    return (5,1,1,2)
  elif style == "-.-":
    return (2,1,2,1)
  elif style == "-.--":
    return (.5,.5,1,.5)
  else:
    return (None, None)

import os
def make_dir(_dir):
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

alg_labels = {"OracleTS": ("OracleTS",  "cyan",  "--"),
              "TS":       ("TS",        "blue",  ":"),
              "AdaTS":    ("B-metaSRM", "red",   "-"),
              "MisAdaTS": ("MisB-metaSRM", "salmon",   "-.--"),
              'mts':      ('f-metaSRM', "green", "-."),
              }

from datetime import datetime
def get_pst_time():
    date_format = '%m_%d_%Y_%H_%M_%S'
    date = datetime.now()
    pstDateTime=date.strftime(date_format)
    return pstDateTime


def savepdfviasvg(fig, name, fig_dir, **kwargs):  # there is a reason for having svg to pdf :)
    # name = ''.join(ch for ch in name if ch not in set(string.punctuation)).replace(" ", "_")
    name = ''.join(ch for ch in name if ch not in "/\\[]\')(,").replace(" ", "_")
    fig.savefig(name + ".svg", format="svg", bbox_inches='tight', pad_inches=0, **kwargs)
    incmd = ["inkscape", name + ".svg", "--export-filename={}/{}.pdf".format(fig_dir, name),
             "--export-pdf-version=1.5"]  # "--export-ignore-filters",
    os.system(' '.join(incmd))
