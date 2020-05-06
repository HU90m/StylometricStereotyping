#!/bin/python3
#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
IMPORT_RESULTS = False
SAVE_RESULTS = True
PLOT_RESULTS = False
SELECTED_MODELS = (
    #'Linear Regression',
    #'Ridge',
    #'Lasso',
    #'RidgeCV',
    #'SVR',
    #'Linear SVR',
    'CatBoost',
)
TESTS = (
    'cv',
    'test',
)
CV_YTICKS = [num/100 - 0.05 for num in range(0, 95, 5)]
TEST_YTICKS = [num/100 for num in range(0, 70, 5)]

#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import evaluate
import relgen

import sys
import os

import numpy as np
import catboost as cat

from scipy import sparse

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.svm import SVR, SVC

if PLOT_RESULTS:
    from matplotlib import pyplot as plt
    from matplotlib import rc
    rc('font',**{'family':'DejaVu Sans','serif':['Times']})
    rc('text', usetex=True)

#---------------------------------------------------------------------------
# Constants
#---------------------------------------------------------------------------
#
cat_params = {
    'verbose': False,
    'task_type': 'GPU',
    'od_type': 'Iter',
    'od_wait': 25,
    'iterations': 1000,
    'depth': 6,
    'l2_leaf_reg': 3,
    'learning_rate': 0.03,
    'bagging_temperature': 1,
}
models = {
    'Linear Regression' : (
        LinearRegression(),
        'linreg',
        'royalblue',
    ),
    'Ridge' : (
        Ridge(),
        'ridge',
        'limegreen',
    ),
    'SVR' : (
        SVR(),
        'svr',
        'darkorange',
    ),
    'Linear SVR' : (
        LinearSVR(),
        'lin_svr',
        'purple',
    ),
    'CatBoost' : (
        cat.CatBoostRegressor(**cat_params),
        'catboost',
        'grey',
    ),
    'Lasso' : (
        Lasso(),
        'lasso',
        'blue',
    ),
    'RidgeCV' : (
        RidgeCV(alphas=[num/100 for num in range(5, 500, 5)]),
        'ridgecv',
        'yellow',
    ),
}
results_prefix = 'results_'
reliabilities_prefix = 'reliabilities_'
test_reliabilities_prefix = 'test-reliabilities_'
distributions = (
    'vvvsimilar',
    'vvsimilar',
    'vsimilar',
    'similar',
    'different',
    'vdifferent',
    'vvdifferent',
    'vvvdifferent',
)
vectors_file = 'reduced_lsa.npy'
test_vectors_file = 'test-reduced_lsa.npy'


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if not os.path.isdir(sys.argv[1]):
            print(f'{sys.argv[1]} is not a directory')
            sys.exit(0)
        else:
            os.chdir(sys.argv[1])

    vectors = np.load(vectors_file, allow_pickle=True)
    test_vectors = np.load(test_vectors_file, allow_pickle=True)


    KLs = []
    for idx, distribution in enumerate(distributions):
        KLs.append(relgen.KLDiv(
                relgen.CATEGORY_BETAS[distribution][0][0],
                relgen.CATEGORY_BETAS[distribution][0][1],
                relgen.CATEGORY_BETAS[distribution][1][0],
                relgen.CATEGORY_BETAS[distribution][1][1],
        ))

    targets = []
    test_targets = []
    for idx, distribution in enumerate(distributions):
        targets.append(np.load(
            reliabilities_prefix + distribution + '.npy',
            allow_pickle=True,
        ))
        test_targets.append(np.load(
            test_reliabilities_prefix + distribution + '.npy',
            allow_pickle=True,
        ))

    results = {}
    if 'cv' in TESTS:
        results['cv'] = {}
        if IMPORT_RESULTS:
            for model_name in SELECTED_MODELS:
                results_file = \
                    results_prefix + 'cv_' + models[model_name][1] + '.npy'
                results['cv'][model_name] = np.load(
                    results_file,
                    allow_pickle=True,
                )
        else:
            for model_name in SELECTED_MODELS:
                results['cv'][model_name] = ([], [])
                jobs = None if model_name == 'CatBoost' else -1
                for idx, distribution in enumerate(distributions):
                    result = evaluate.cross_validate_model(
                        models[model_name][0],
                        vectors,
                        targets[idx],
                        is_classification=False,
                        splits=5,
                        jobs=jobs,
                    )
                    results['cv'][model_name][0].append(result[0])
                    results['cv'][model_name][1].append(result[1])

                if SAVE_RESULTS:
                    results_file = \
                        results_prefix + 'cv_' + models[model_name][1] + '.npy'
                    np.save(results_file, results['cv'][model_name])

    if 'test' in TESTS:
        results['test'] = {}
        if IMPORT_RESULTS:
            for model_name in SELECTED_MODELS:
                results_file = \
                    results_prefix + 'test_' + models[model_name][1] + '.npy'
                results['test'][model_name] = np.load(
                    results_file,
                    allow_pickle=True,
                )
        else:
            for model_name in SELECTED_MODELS:
                results['test'][model_name] = ([], [])
                for idx, distribution in enumerate(distributions):
                    result = evaluate.test_model(
                        models[model_name][0],
                        vectors,
                        targets[idx],
                        test_vectors,
                        test_targets[idx],
                        is_classification=False,
                    )
                    results['test'][model_name][0].append(result[0])
                    results['test'][model_name][1].append(result[1])

                if SAVE_RESULTS:
                    results_file = \
                        results_prefix +'test_'+ models[model_name][1] + '.npy'
                    np.save(results_file, results['test'][model_name])


    if PLOT_RESULTS:
        def plot_test(plot_r2, plot_rmse, test):
            for model_name in SELECTED_MODELS:
                plot_r2.plot(KLs,
                    results[test][model_name][0],
                    label=model_name,
                    color=models[model_name][2],
                    linewidth=1,
                    linestyle='-',
                    marker='|',
                    markersize=3,
                )
                plot_rmse.plot(KLs,
                    results[test][model_name][1],
                    label=model_name,
                    color=models[model_name][2],
                    linewidth=1,
                    linestyle='-',
                    marker='|',
                    markersize=3,
                )
            plot_rmse.set_ylabel('$RMSE$')
            plot_r2.set_ylabel('$R^2$')
            plot_r2.set_xticks(range(10))
            plot_rmse.set_xticks(range(10))
            if test == 'cv':
                plot_r2.set_title('5 Fold Cross Validation')
            else:
                plot_r2.set_title('Test Set Performance')
            plot_rmse.set_xlabel('$D_{KL}$')
            #plot_r2.set_yticks(CV_YTICKS)
            plot_r2.grid('both')
            plot_rmse.grid('both')

        fig = plt.figure()
        ax11 = plt.subplot(221)
        ax21 = plt.subplot(223, sharex=ax11)

        if len(TESTS) > 1:
            ax12 = plt.subplot(222, sharey=ax11)
            ax22 = plt.subplot(224, sharey=ax21, sharex=ax12)

            plot_test(ax11, ax21, 'test')
            plot_test(ax12, ax22, 'cv')

        else:
            plot_test(ax11, ax12, TESTS[0])

        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.show()

    print('All Done.')
