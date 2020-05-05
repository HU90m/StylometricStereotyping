#!/bin/python3
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
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    OP_CV = True
    cat_params = {
        "iterations": 100,
        "depth": 6,
        "verbose": False,
        "task_type": "GPU",
    }
    models = {
        'Linear Regression' : LinearRegression(),
        #'Lasso' : Lasso(),
        'Ridge' : Ridge(),
        'RidgeOther' : RidgeCV(alphas=[num/100 for num in range(5, 500, 5)]),
        #'SVR' : SVR(),
        #'SGD' : SGDRegressor(),
        #'Linear SVR' : LinearSVR(),
        #'CatBoost' : cat.CatBoostRegressor(**cat_params),
    }
    colours = {
        'Linear Regression' : 'red',
        'Lasso' : 'blue',
        'Ridge' : 'blue',
        'RidgeOther' : 'green',
        'SVR' : 'yellow',
        'SGD' : 'cyan',
        'Linear SVR' : 'magenta',
        'CatBoost' : 'cyan',
    }
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

    vectors = np.load('reduced_lsa.npy', allow_pickle=True)
    test_vectors = np.load('test-reduced_lsa.npy', allow_pickle=True)


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
    for model_name in models:
        results[model_name] = []
        for idx, distribution in enumerate(distributions):
            if OP_CV:
                results[model_name].append(evaluate.cross_validate_model(
                    models[model_name],
                    vectors,
                    targets[idx],
                    is_classification=False,
                    splits=2,
                ))
            else:
                results[model_name].append(evaluate.test_model(
                    models[model_name],
                    vectors,
                    targets[idx],
                    test_vectors,
                    test_targets[idx],
                    is_classification=False,
                )[0])


    for model_name in models:
        plt.plot(KLs,
            results[model_name],
            label=model_name,
            color=colours[model_name],
            linewidth=1,
            linestyle='-',
        )

    plt.xlabel('$D_{KL}$')
    plt.xticks(range(10))
    plt.ylabel('$R^2$')
    if OP_CV:
        plt.yticks([num/100 - 0.05 for num in range(0, 95, 5)])
    else:
        plt.yticks([num/100 for num in range(0, 70, 5)])
    plt.grid('both')
    plt.legend()
    plt.show()

    print('All Done.')
