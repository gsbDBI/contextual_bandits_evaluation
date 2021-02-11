"""
This script contains helping functions of different saving formats. 
"""

import numpy as np
import pandas as pd
import os
import subprocess
from time import time
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser


__all__ = [
    "compose_filename",
    "get_sherlock_dir",
    "misc_to_pandas",
    "on_sherlock",
    "project_dir",
    "stats_to_pandas",

]

__file__ = os.getcwd()
project_dir = dirname(dirname(realpath(__file__)))


def on_sherlock():
    """ Checks if running locally or on sherlock """
    return 'GROUP_SCRATCH' in os.environ


def get_sherlock_dir(project, *tail, create=True):
    """
    Output consistent folder name in Sherlock.
    If create=True and on Sherlock, also makes folder with group permissions.
    If create=True and not on Sherlock, does not create anything.

    '/scratch/groups/athey/username/project/tail1/tail2/.../tailn'.

    >>> get_sherlock_dir('adaptive-inference')
    '/scratch/groups/athey/adaptive-inference/vitorh'

    >>> get_sherlock_dir('toronto')
    '/scratch/groups/athey/toronto/vitorh/'

    >>> get_sherlock_dir('adaptive-inference', 'experiments', 'exp_out')
    '/scratch/groups/athey/adaptive-inference/vitorh/experiments/exp_out'
    """
    base = join("/", "scratch", "groups", "athey", getuser(), project)
    path = join(base, *tail)
    if not exists(path) and create and on_sherlock():
        makedirs(path, exist_ok=True)
        # Correct permissions for the whole directory branch
        chmod_path = base
        chmod(base, 0o775)
        for child in tail:
            chmod_path = join(chmod_path, child)
            chmod(chmod_path, 0o775)
    return path


def compose_filename(prefix, extension):
    """
    Creates a unique filename.
    Useful when running in parallel on Sherlock.
    """
    # Tries to find a commit hash
    try:
        commit = subprocess\
            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time() * 1e8 % 1e8))
    if on_sherlock():
        sid = os.environ['SLURM_JOB_ID']
        tid = os.environ['SLURM_LOCALID']
        jid = os.environ['SLURM_JOB_NAME']
    else:
        sid = tid = jid = ''
    ident = filter(None, [prefix, commit, jid, sid, tid, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname


def stats_to_pandas(stats, timepoints, **prefix):
    cols = ["estimate", "stderr", "bias", "coverage", "relerror", "tstat", "power", "mse"]
    S, _, K = stats.shape
    tps = len(timepoints)
    names = np.repeat(cols, tps * K)
    times = np.repeat(np.tile(timepoints, S), K)
    policy = np.tile(np.arange(K), tps * S)
    values = np.ravel(stats[:, timepoints, :])
    df = pd.DataFrame(dict(**prefix, policy=policy, time=times,
                           statistic=names, value=values))
    return df


def misc_to_pandas(other, name, timepoints, **prefix):
    tps = len(timepoints)
    if np.ndim(other) == 1:
        K = 1
        policy = np.nan
    else:
        _, K = other.shape
        policy = np.tile(np.arange(K), tps)
    names = np.repeat([name], tps * K)
    times = np.repeat(timepoints, K)
    values = np.ravel(other[timepoints])
    df = pd.DataFrame(dict(**prefix, policy=policy, time=times,
                           statistic=names, value=values))
    return df
