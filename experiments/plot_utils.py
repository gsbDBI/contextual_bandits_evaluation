"""
This script contains helper functions to make plots presented in the paper 
"""
from itertools import product
from itertools import compress
import copy
from pickle import UnpicklingError
import dill as pickle
from adaptive.saving import *
from IPython.display import display, HTML
import scipy.stats as stats
from glob import glob
from time import time
from scipy.stats import norm
import seaborn as sns
from adaptive.compute import collect
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.ticker import FormatStrFormatter

np.seterr(all='raise')


def read_files(file_name):
    files = glob(file_name)
    print(f'Found {len(files)} files.')
    results = []
    for file in files:
        try:
            with open(file, 'rb') as f:
                r = pickle.load(f)
            results.extend(r)
        except:  # UnpicklingError:
            print(f"Skipping corrupted file: {file}")
    return results


def add_config(dfs, r):
    dfs = pd.concat(dfs)
    for key in r['config']:
        if key == 'policy_names':
            continue
        dfs[key] = r['config'][key]
    return dfs


def save_data_timepoints(data, timepoints, method, K, order):
    data = data[timepoints, :]
    return pd.DataFrame({
        "time": np.tile(timepoints, K),
        "policy": np.repeat(np.arange(K), len(timepoints)),
        "value": data.flatten(order=order),
        "method": [method] * data.size,
    })


def generate_data_frames(results):
    """
    Generate DataFrames from the raw saving results.
    """
    df_stats = []
    df_probs = []
    df_covs = []
    for r in results:

        CONFIG_COLS = list(r['config'].keys())
        CONFIG_COLS.remove('policy_value')

        # get statistics table
        tabs_stats = []
        T = r['config']['T']
        for weight, stats in r['stats'].items():
            statistics = ['Bias', 'Var']
            tab_stat = pd.DataFrame({"statistic": statistics,
                                     "value": stats.flatten(),
                                     'weight': [weight] * len(statistics)
                                     })
            tabs_stats.append(tab_stat)

        df_stats.append(add_config(tabs_stats, r))

    df_stats = pd.concat(df_stats)

    # add true standard error, relative variance, relerrors and coverage in df_stats
    confidence_level = np.array([0.9, 0.95])
    quantile = norm.ppf(0.5+confidence_level/2)
    new_stats = []
    # group_keys = [*CONFIG_COLS, 'policy', 'weight',]
    group_keys = ['experiment', 'policy', 'weight']
    for *config, df_cfg in df_stats.groupby(group_keys):
        weight = config[0][group_keys.index('weight')]
        df_bias = df_cfg.query("statistic=='Bias'")
        df_var = df_cfg.query("statistic=='Var'")
        true_se = np.std(df_bias['value'])

        if true_se < 1e-6:
            print(
                f"For config {dict(zip([*CONFIG_COLS, 'policy', 'weight'], config))} data is not sufficient, only has {len(df_bias)} samples.")
            continue

        # relative S.E.
        df_relse = pd.DataFrame.copy(df_var)
        df_relse['value'] = np.sqrt(np.array(df_relse['value'])) / true_se
        df_relse['statistic'] = 'relative S.E.'

        # true S.E.
        df_truese = pd.DataFrame.copy(df_var)
        df_truese['value'] = true_se
        df_truese['statistic'] = 'true S.E.'

        # relative error
        df_relerror = pd.DataFrame.copy(df_bias)
        df_relerror['value'] = np.array(df_relerror['value']) / true_se
        df_relerror['statistic'] = 'R.E.'

        # tstat
        df_tstat = pd.DataFrame.copy(df_bias)
        df_tstat['value'] = np.array(
            df_tstat['value']) / np.sqrt(np.array(df_var['value']))
        df_tstat['statistic'] = 't-stat'

        new_stats.extend([df_relse, df_truese, df_relerror, df_tstat])

        # coverage

        for p, q in zip(confidence_level, quantile):
            df_relerror_cov = pd.DataFrame.copy(df_relerror)
            df_relerror_cov['value'] = (
                np.abs(np.array(df_relerror['value'])) < q).astype(float)
            df_relerror_cov['statistic'] = f'{int(p*100)}% coverage of R.E.'

            df_tstat_cov = pd.DataFrame.copy(df_tstat)
            df_tstat_cov['value'] = (
                np.abs(np.array(df_tstat_cov['value'])) < q).astype(float)
            df_tstat_cov['statistic'] = f'{int(p*100)}% coverage of t-stat'

            df_covs.extend([df_relerror_cov, df_tstat_cov])

    df_stats = pd.concat([df_stats, *new_stats, *df_covs])

    return df_stats, CONFIG_COLS, confidence_level


def plot_statistics(df_stats, row='policy', name=None, order=None, order_name=None):
    """
    Plot statistics of different estimates across time. 
    """
    col = 'statistic'
    col_order = ['Bias', 'Bias', 'Var',  '90% coverage of t-stat']
    confidence = float(col_order[-1][:2])/100
    if row == 'policy':
        row_order = ['optimal-best_arm', 'optimal', 'best_arm']
    else:
        row_order = ['Signal', 'No Signal']

    hue = 'weight'
    hue_order = df_stats[hue].unique()
    order = ['uniform', 'propscore_expected',
             'propscore_X', 'lvdl_expected', 'lvdl_X', 'DM']
    order_2 = ['uniform', 'propscore_expected',
               'propscore_X', 'lvdl_expected', 'lvdl_X']
    sns_palette = sns.color_palette()
    palette = [sns_palette[0], sns_palette[1], sns_palette[1],
               sns_palette[2], sns_palette[2], sns_palette[3]]
    linestyles = ['-', '-', '--', '-', '--', '-']
    g = sns.catplot(x='T',
                    y="value",
                    col=col,
                    col_order=col_order,
                    row=row,
                    row_order=row_order,
                    kind="point",
                    hue=hue,
                    hue_order=order,
                    palette=palette,
                    aspect=1.2,
                    # markers=markers,
                    linestyles=linestyles,
                    sharex=False,
                    sharey=False,
                    legend=False,
                    legend_out=True,
                    margin_titles=True,
                    data=df_stats)

    for i in range(len(row_order)):
        g.axes[i, 0].clear()
        sns.pointplot(x='T',
                      # order=order,
                      hue=hue,
                      hue_order=order,
                      palette=palette,
                      linestyles=linestyles,
                      y="value",
                      ax=g.axes[i, 0],
                      data=df_stats.query(
                          f"{row}=='{row_order[i]}' & statistic=='Bias'"),
                      estimator=lambda x: np.sqrt(np.mean(x**2)),
                      )
        g.axes[i, 0].legend("")
        g.axes[i, 0].set_xlabel("")
        g.axes[i, 0].set_ylabel("")

        g.axes[i, 1].axhline(0.0, color="black", linestyle='-.')

        g.axes[i, 2].clear()
        sns.pointplot(x='T',
                      # order=order,
                      hue=hue,
                      hue_order=order_2,
                      palette=palette,
                      linestyles=linestyles,
                      y="value",
                      ax=g.axes[i, 2],
                      data=df_stats.query(
                          f"{row}=='{row_order[i]}' & statistic=='Var'"),
                      estimator=lambda x: np.mean(
                          np.sqrt(x) * norm.ppf(confidence/2+0.5)),
                      )
        g.axes[i, 2].legend("")
        g.axes[i, 2].set_xlabel("")
        g.axes[i, 2].set_ylabel("")

        g.axes[i, -1].clear()
        sns.pointplot(x='T',
                      # order=order,
                      hue=hue,
                      hue_order=order_2,
                      palette=palette,
                      linestyles=linestyles,
                      y="value",
                      ax=g.axes[i, -1],
                      data=df_stats.query(
                          f"{row}=='{row_order[i]}' & statistic=='{col_order[-1]}'"),
                      )
        g.axes[i, -1].legend("")
        g.axes[i, -1].set_xlabel("")
        g.axes[i, -1].set_ylabel("")

        g.axes[i, -1].axhline(confidence, color="black", linestyle='-.')
        g.axes[i, -1].set_ylim(0.80, 0.93)

    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
        #ax.set_xticklabels(['DM', 'uniform', 'props\nexpected', 'props\nX', 'stabvar\nexpected', 'stabvar\nX'])
    g.col_names = ['RMSE', 'Bias', 'Confidence Interval Radius', col_order[-1]]
    if row == 'experiment':
        g.row_names = ['Signal', 'No Signal']
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    g.set_xlabels("")
    g.set_ylabels("")
    # g.set_xticklabels(order,rotation=270)

    handles, labels = g._legend_data.values(), g._legend_data.keys()
    new_handles = [Line2D([0], [0], color=pal, linewidth=4, linestyle="--" if (
        k == 2 or k == 4) else "-") for k, pal in enumerate(palette)]
    names = dict(uniform='DR', DM='DM',
                 propscore_expected='non-contextual MinVar', propscore_X='contextual MinVar',
                 lvdl_expected='non-contextual StableVar', lvdl_X='contextual StableVar',
                 )
    g.fig.legend(labels=[names[n] for n in labels], handles=new_handles,
                 loc='center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.005))

    # g.fig.tight_layout()

    if name is not None:
        # g.fig.suptitle(f'{name} statistics', x=0.5, y=0.0, fontsize=20)
        g.savefig(f'figures/{name}.eps',  bbox_inches="tight")
    plt.show()


def analyze_characteristics(df_stats):
    """
    Analyze dataset characteristics including #features, #classes, #observations.
    """
    classes = dict()
    features = dict()
    observations = dict()
    for dgp, df_dgp in df_stats.groupby('dgp'):
        K = int(np.mean(df_dgp['K']))
        T = int(np.mean(df_dgp['T']))
        p = int(np.mean(df_dgp['p']))
        if K not in classes:
            classes[K] = 0
        classes[K] += 1
        if p not in features:
            features[p] = 0
        features[p] += 1
        if T not in observations:
            observations[T] = 0
        observations[T] += 1
    sorted(zip(classes.keys(), classes.values()), key=lambda x: x[0])
    a = [0, 0, 0]
    for c in classes:
        if c == 2:
            a[0] += classes[c]
        elif c <= 8:
            a[1] += classes[c]
        else:
            a[2] += classes[c]
    print(f"#Classes: =2: {a[0]}; >2,<=8: {a[1]}; >8: {a[2]}")

    sorted(zip(features.keys(), features.values()), key=lambda x: x[0])
    a = [0, 0, 0]
    for k in features:
        if k <= 10:
            a[0] += features[k]
        elif k <= 50:
            a[1] += features[k]
        else:
            a[2] += features[k]
    print(f"#Features: <=10: {a[0]}; >10, <=50: {a[1]}; >50: {a[2]} ")

    sorted(zip(observations.keys(), observations.values()), key=lambda x: x[0])
    a = [0, 0, 0]
    for k in observations:
        if k <= 1000:
            a[0] += observations[k]
        elif k <= 10000:
            a[1] += observations[k]
        else:
            a[2] += observations[k]
    print(f"#Observations: <=1k: {a[0]}; >1k, <=10k: {a[1]}; >10k: {a[2]}")


def merge_dataset_df(df_stats):
    """
    Merge statistics of one dataset across different simulations.
    """
    dgp = []
    bias = []
    se = []
    policy = []
    weight = []
    K = []
    p = []
    T = []
    for config, df_dgp in df_stats.groupby(['dgp', 'policy', 'weight']):
        df_bias = pd.DataFrame.copy(df_dgp.query("statistic=='Bias'"))
        df_se = pd.DataFrame.copy(df_dgp.query("statistic=='true S.E.'"))
        dgp.append(config[0])
        policy.append(config[1])
        K.append(np.mean(df_dgp['K']))
        p.append(np.mean(df_dgp['p']))
        T.append(np.mean(df_dgp['T']))
        weight.append(config[2])
        bias.append(np.mean(df_bias['value']))
        se.append(np.mean(df_se['value']))
    return pd.DataFrame(dict(dgp=dgp, bias=bias, se=se, policy=policy, weight=weight, T=T, K=K, p=p))


def get_baseline(df_stats, weight, policy):
    bias = np.array(df_stats.query(
        f"weight=='{weight}' & policy=='{policy}' ")['bias'])
    std = np.array(df_stats.query(
        f"weight=='{weight}' & policy=='{policy}' ")['se'])
    baseline = np.sqrt(bias**2 + std**2)
    return baseline


def plot_radius(df_stats, weight, policy):
    """
    Use one weighting method as a baseline, and plot others (bias, S.E.) normalized by the RMSE of the baseline.
    """
    baseline = get_baseline(df_stats, weight, policy)
    f, ax = plt.subplots(nrows=2, ncols=3,  figsize=(
        3*6, 2*6), subplot_kw=dict(polar=True))
    names = dict(propscore_X='contextual MinVar', propscore_expected='non-contextual MinVar',
                 lvdl_expected='non-contextual StableVar', lvdl_X='contextual StableVar',
                 DM='DM', uniform='DR')
    for c, (w, df_w) in enumerate(df_stats.query(f"policy=='{policy}'").groupby('weight')):
        i, j = c // 3, c % 3
        bias_w = np.array(df_w['bias'])
        std_w = np.array(df_w['se'])
        theta = np.pi/2 - np.arctan(bias_w / std_w)
        r = np.sqrt(bias_w ** 2 + std_w ** 2) / baseline
        ax[i, j].plot(theta, r, '.', c='darkorange', alpha=0.7)
        ax[i, j].plot(np.linspace(0, np.pi, 50), [1]*50, 'k--', alpha=0.7)
        ax[i, j].fill_between(np.linspace(
            np.pi/4, 3*np.pi/4, 50), 0, 3, alpha=0.2, color='c')

        ax[i, j].set_thetamin(0)
        ax[i, j].set_thetamax(180)
        ax[i, j].set_rmax(3)
        ax[i, j].set_title(f'{names[w]}', y=0.85, fontsize=20)
        ax[i, j].set_rticks([0, 1, 2])  # less radial ticks
        ax[i, j].set_xticks([0, np.pi/2, np.pi])
        ax[i, j].set_xticklabels(['', "S.E.", 'Bias'])

    plt.subplots_adjust(hspace=-0.45)


def plot_radius_comparison(df_stats, policy):
    """
    Compare contextual weighting and non-contextual weighting.
    """
    f, ax = plt.subplots(nrows=1, ncols=2,  figsize=(
        2*6, 1*6), subplot_kw=dict(polar=True))

    def plot_polar_ax(weight, baseline_weight, policy, ax, name):
        baseline = get_baseline(df_stats, baseline_weight, policy)
        bias_w = np.array(df_stats.query(
            f" weight=='{weight}' & policy=='{policy}'")['bias'])
        std_w = np.array(df_stats.query(
            f" weight=='{weight}' & policy=='{policy}'")['se'])
        theta = np.pi/2 - np.arctan(bias_w / std_w)
        r = np.sqrt(bias_w ** 2 + std_w ** 2) / baseline
        ax.plot(theta, r, '.', c='darkorange', alpha=0.7)
        ax.plot(np.linspace(0, np.pi, 50), [1]*50, 'k--', alpha=0.7)
        ax.fill_between(np.linspace(np.pi/4, 3*np.pi/4, 50),
                        0, 2, alpha=0.2, color='c')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rmax(2)
        ax.set_title(name, y=0.85, fontsize=15)
        ax.set_rticks([0, 1, 2])  # less radial ticks
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['', "S.E.", 'Bias'])
    plot_polar_ax('propscore_X', 'propscore_expected', policy,
                  ax[0], 'contextual MinVar / non-contextual MinVar')
    plot_polar_ax('lvdl_X', 'lvdl_expected', policy,
                  ax[1], 'contextual StableVar / non-contextual StableVar')


def plot_radius_comparison_2(df_stats, policy):
    """
    Compare MinVar and StableVar.
    """
    f, ax = plt.subplots(nrows=1, ncols=2,  figsize=(
        2*6, 1*6), subplot_kw=dict(polar=True))

    def plot_polar_ax(weight, baseline_weight, policy, ax, name):
        baseline = get_baseline(df_stats, baseline_weight, policy)
        bias_w = np.array(df_stats.query(
            f" weight=='{weight}' & policy=='{policy}'")['bias'])
        std_w = np.array(df_stats.query(
            f" weight=='{weight}' & policy=='{policy}'")['se'])
        theta = np.pi/2 - np.arctan(bias_w / std_w)
        r = np.sqrt(bias_w ** 2 + std_w ** 2) / baseline
        ax.plot(theta, r, '.', c='darkorange', alpha=0.7)
        ax.plot(np.linspace(0, np.pi, 50), [1]*50, 'k--', alpha=0.7)
        ax.fill_between(np.linspace(np.pi/4, 3*np.pi/4, 50),
                        0, 2, alpha=0.2, color='c')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rmax(2)
        ax.set_title(name, y=0.85, fontsize=15)
        ax.set_rticks([0, 1, 2])  # less radial ticks
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['', "S.E.", 'Bias'])
    plot_polar_ax('propscore_expected', 'lvdl_expected', policy,
                  ax[0], 'non-contextual MinVar / non-contextual StableVar')
    plot_polar_ax('propscore_X', 'lvdl_X', policy,
                  ax[1], 'contextual MinVar / contextual StableVar')
