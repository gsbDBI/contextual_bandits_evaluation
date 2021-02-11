"""This script runs simulations on synthetic dataset."""
import sys
from adaptive.inference import analyze_by_continuous_X, aw_scores
import argparse
import os
import pickle
from time import time
from adaptive.experiment import *
from adaptive.ridge import *
from adaptive.datagen import *
from adaptive.saving import *


parser = argparse.ArgumentParser(description='Process DGP settings')
parser.add_argument(
    '-n',
    '--name',
    type=str,
    default='synthetic',
    help='saving name of experiments')
parser.add_argument(
    '--floor_decay',
    type=float,
    default=0.8,
    help='assignment probability floor decay')
parser.add_argument(
    '--split',
    type=float,
    default=0.5,
    help='boundary split')
parser.add_argument(
    '-b',
    '--bandit_model',
    type=str,
    default='RegionModel',
    help='bandit model')
parser.add_argument(
    '--noise_form',
    type=str,
    default='normal',
    help='noise function')
parser.add_argument(
    '-s',
    '--sims',
    type=int,
    default=1,
    help='number of simulations')
parser.add_argument(
    '--signal',
    type=float,
    default=0.5,
    help='signal strength')
parser.add_argument(
    '-T',
    '--T',
    type=int,
    default=7000,
    help='sample size')


if __name__ == '__main__':

    t1 = time()

    """ Experiment configuration """
    args = parser.parse_args()
    num_sims = args.sims
    save_every = 20


    """ Run and analyze the experiment """
    results_list = []
    for s in range(num_sims):
        # configs
        floor_decay = args.floor_decay
        bandit_model = args.bandit_model 
        K = 4
        p = 3
        T = args.T
        config = {
            'T': T,
            'K': K,
            'p': p,
            'noise_std': 1,
            'signal': args.signal, 
            'experiment': args.name,
            'dgp': 'tree',
            'split': args.split,
            'floor_start': 1 / K,
            'floor_decay': floor_decay,
            'noise_form': args.noise_form,
            'bandit_model': bandit_model,
            'batch_size': 100,
            'time_explore': 50 * K,

        }
        batch_sizes = [config['time_explore']] + [config['batch_size']
                        ] * int((T - config['time_explore'])/config['batch_size'])
        if np.sum(batch_sizes) < T:
            batch_sizes[-1] += T - np.sum(batch_sizes)

        if (s+1) % 20 == 0:
            print(f'Simulation {s+1}/{num_sims}.')

        """ Data generation """
        # Collect data from environment
        data_exp, mus = simple_tree_data(
            T=T, K=K, p=p, noise_std=config['noise_std'], 
                split=config['split'], signal_strength=args.signal,
            noise_form=args.noise_form)
        xs, ys = data_exp['xs'], data_exp['ys']

        # Run the contextual bandits experiment on the simulated data
        data = run_experiment(xs, ys, config, batch_sizes=batch_sizes)
        yobs, ws, probs = data['yobs'], data['ws'], data['probs']

        """ Target policies """
        policy_names = ['random', 'optimal', 'best_arm']
        policy_values = [np.mean(mus), args.signal, max(mus)]

        policy_mtx = []
        # add random policy
        policy_mtx.append(np.ones((T, K)) / K)  

        # add optimal policy
        policy_mtx.append(data_exp['wxs'])

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
        

        for Tt in [1000, 3000, 5000, 7000]:
            muhat_DM = ridge_muhat_DM(xs[:Tt], ws[:Tt], yobs[:Tt], K)
            for policy_m, policy_v, policy_n in zip(policy_mtx, policy_values, policy_names):
                analysis = analyze_by_continuous_X(
                    probs=probs[:Tt, :Tt],
                    gammahat=gammahat[:Tt],
                    policy=policy_m[:Tt],
                    policy_value=policy_v,
                )
                DM_estimate = np.sum(policy_m[:Tt] * muhat_DM, 1)
                DM_estimate_var = np.sum((yobs[:Tt] - np.mean(DM_estimate))**2) / (Tt ** 2)
                analysis['DM'] = np.array([np.mean(DM_estimate)-policy_v,
                    DM_estimate_var])
                new_config = deepcopy(config)
                new_config['policy'] = policy_n
                new_config['policy_value'] = policy_v
                new_config['T'] = Tt
                results = {'stats': analysis, 'config': new_config}
                results_list.append(results)

        """ Saving results """
        if (s+1) % save_every == 0 or s == num_sims-1:
            if on_sherlock():
                experiment_dir = get_sherlock_dir('aw_contextual')
            else:
                experiment_dir = os.path.join(os.getcwd(), 'results')
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)

            filename = compose_filename(f'{config["experiment"]}_{s}', 'pkl')
            write_path = os.path.join(experiment_dir, filename)
            print(f"Saving at {write_path}")
            with open(write_path, 'wb') as f:
                pickle.dump(results_list, f)
            results_list = []

    print(f'Running time {time() - t1}s')
