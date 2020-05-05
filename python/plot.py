#!/bin/python3
#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
IMPORT_RESULTS = False
SAVE_RESULTS = True
PLOT_RESULTS = True
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
from os import path

import numpy as np
import catboost as cat
from matplotlib import pyplot as plt

from scipy import sparse

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.svm import SVR, SVC


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
        'red',
    ),
    'Ridge' : (
        Ridge(),
        'ridge',
        'green',
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
    'SVR' : (
        SVR(),
        'svr',
        'cyan',
    ),
    'Linear SVR' : (
        LinearSVR(),
        'lin_svr',
        'magenta',
    ),
    'CatBoost' : (
        cat.CatBoostRegressor(**cat_params),
        'catboost',
        'grey',
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
                results['cv'][model_name] = []
                jobs = None if model_name == 'catboost' else -1
                for idx, distribution in enumerate(distributions):
                    results['cv'][model_name].append(
                        evaluate.cross_validate_model(
                            models[model_name][0],
                            vectors,
                            targets[idx],
                            is_classification=False,
                            splits=5,
                            jobs=jobs,
                        )
                    )

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
                results['test'][model_name] = []
                for idx, distribution in enumerate(distributions):
                    results['test'][model_name].append(evaluate.test_model(
                        models[model_name][0],
                        vectors,
                        targets[idx],
                        test_vectors,
                        test_targets[idx],
                        is_classification=False,
                    )[0])

                if SAVE_RESULTS:
                    results_file = \
                        results_prefix +'test_'+ models[model_name][1] + '.npy'
                    np.save(results_file, results['test'][model_name])


    if PLOT_RESULTS:
        fig, plots = plt.subplots(ncols=len(TESTS), nrows=1)

        def plot_test(plot, test):
            for model_name in SELECTED_MODELS:
                plot.plot(KLs,
                    results[test][model_name],
                    label=model_name,
                    color=models[model_name][2],
                    linewidth=1,
                    linestyle='-',
                )
            plot.set_xlabel('$D_{KL}$')
            plot.set_xticks(range(10))
            plot.set_ylabel('$R^2$')
            if test == 'cv':
                plot.set_title('5 Fold Cross Validation')
                plot.set_yticks(CV_YTICKS)
            else:
                plot.set_title('Test Set Performance')
                plot.set_yticks(TEST_YTICKS)
            plot.grid('both')

        if len(TESTS) > 1:
            for idx, test in enumerate(TESTS):
                plot_test(plots[idx], test)
        else:
            plot_test(plots, TESTS[0])

        plt.legend()
        plt.tight_layout()
        plt.show()

    print('All Done.')
