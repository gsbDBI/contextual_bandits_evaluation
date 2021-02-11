import sys
from scipy.stats import multivariate_normal
from adaptive.inference import analyze_by_continuous_X, aw_scores
import argparse
import os
import pickle
from time import time
from adaptive.experiment import *
from adaptive.ridge import *
from adaptive.datagen import *
from adaptive.saving import *
from glob import glob
from copy import deepcopy
import openml

parser = argparse.ArgumentParser(description='Process DGP settings')
parser.add_argument(
    '-n',
    '--name',
    type=str,
    default='classification',
    help='saving name of experiments')
parser.add_argument(
    '--floor_decay',
    type=float,
    default=0.5,
    help='assignment probability floor decay')
parser.add_argument(
    '-f',
    '--file_name',
    type=str,
    default='yeast',
    help='file name')
parser.add_argument(
    '--signal',
    type=float,
    default=1.0,
    help='signal strength')
parser.add_argument(
    '-s',
    '--sim',
    type=int,
    default=100,
    help='simulation of running one experiment')


if __name__ == '__main__':

    t1 = time()

    """ Experiment configuration """
    project_dir = os.getcwd()
    args = parser.parse_args()

    save_every = 1 
    noise_std = 1.0
    signal_strength = args.signal
    floor_decay = args.floor_decay

    """ Load data sets """
    
    dname = args.file_name
    openml_list = openml.datasets.list_datasets()
    dataset = openml.datasets.get_dataset(dname)
    X, y, _, _ = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)


    """ Run and analyze the experiment """
    results_list = []
    sim = args.sim
    for sim_n in range(sim):
        data_exp, mus = generate_bandit_data(X=X, y=y, noise_std=noise_std, signal_strength=signal_strength)
        bandit_model = 'TSModel'
        xs, ys = data_exp['xs'], data_exp['ys']
        K, p, T = data_exp['K'], data_exp['p'], data_exp['T']
        batch_size = min(100, T//10)
        config = {
            'T': T,
            'K': K,
            'p': p,
            'noise_std': noise_std,
            'signal': signal_strength,
            'experiment': args.name,
            'dgp': dname,
            'floor_start': 1 / K,
            'floor_decay': floor_decay,
            'bandit_model': bandit_model,
            'batch_size': batch_size,
            'time_explore': batch_size//2 * K,

        }
        batch_sizes = [config['time_explore']] + [config['batch_size']
                ] * int((T - config['time_explore'])/config['batch_size'])
        if np.sum(batch_sizes) < T:
            batch_sizes[-1] += T - np.sum(batch_sizes)


        """ Data generation """
        # Run the experiment on the simulated data
        data = run_experiment(xs, ys, config, batch_sizes=batch_sizes)
        yobs, ws, probs = data['yobs'], data['ws'], data['probs']

        """ Evaluated policies """
        policy_names = ['random', 'optimal', 'best_arm']
        policy_values = [np.mean(mus), signal_strength, max(mus)]

        policy_mtx = []
        # add random policy
        policy_mtx.append(np.ones((T, K)) / K)  

        # add optimal policy
        policy_mtx.append(expand(np.ones(T), np.argmax(data_exp['muxs'], axis=1), K))

        # add best arm policy
        best_mtx = np.zeros((T, K))
        best_mtx[:, np.argmax(mus)] = 1
        policy_mtx.append(best_mtx)  

        # add contrast
        policy_names.append('optimal-best_arm')
        policy_mtx.append(policy_mtx[1] - policy_mtx[2])
        policy_values.append(policy_values[1] - policy_values[2])

        """ Evaluation """
        # Estimate muhat and gammahat
        muhat = ridge_muhat_lfo_pai(data_exp['xs'], ws, yobs, K, batch_sizes)
        balwts = 1 / collect(collect3(probs), ws)
        gammahat = aw_scores(yobs=yobs, ws=ws, balwts=balwts,
                             K=K, muhat=collect3(muhat))

        # Estimate muhat_DM
        muhat_DM = ridge_muhat_DM(data_exp['xs'], ws, yobs, K)

        for policy_m, policy_v, policy_n in zip(policy_mtx, policy_values, policy_names):
            analysis = analyze_by_continuous_X(
                probs=probs,
                gammahat=gammahat,
                policy=policy_m,
                policy_value=policy_v,
            )
            estimate_DM = np.sum(policy_m * muhat_DM, axis=1)
            analysis['DM'] = np.array([np.mean(estimate_DM)-policy_v, np.var(estimate_DM)/T])
            config['policy'] = policy_n
            config['policy_value'] = policy_v
            results = {'stats': analysis, 'config': deepcopy(config)}
            results_list.append(results)

        """ Save """
        if sim_n % 10 == 0 or sim_n == sim-1:
            if on_sherlock():
                experiment_dir = get_sherlock_dir('contextual-aipwlfo')
            else:
                experiment_dir = os.path.join(project_dir, 'results')
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)

            filename = compose_filename(f'{config["experiment"]}_{dname}', 'pkl')
            write_path = os.path.join(experiment_dir, filename)
            print(f"Saving at {write_path}")
            with open(write_path, 'wb') as f:
                pickle.dump(results_list, f)
            results_list = []

    print(f'Running time {time() - t1}s')
