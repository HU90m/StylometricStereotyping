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

from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.svm import SVR, SVC


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    cat_params = {
        "iterations": 100,
        "depth": 6,
        "verbose": False,
        "task_type": "GPU",
    }
    models = {
        'Lasso' : Lasso(),
        'Ridge' : Ridge(),
        'SVR' : SVR(),
        'Linear SVR' : LinearSVR(),
        'CatBoost' : cat.CatBoostRegressor(**cat_params),
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
    results = []
    for idx, distribution in enumerate(distributions):
        targets = np.load(
            reliabilities_prefix + distribution + '.npy',
            allow_pickle=True,
        )
        test_targets = np.load(
            test_reliabilities_prefix + distribution + '.npy',
            allow_pickle=True,
        )

        KLs.append(relgen.KLDiv(
                relgen.CATEGORY_BETAS[distribution][0][0],
                relgen.CATEGORY_BETAS[distribution][0][1],
                relgen.CATEGORY_BETAS[distribution][1][0],
                relgen.CATEGORY_BETAS[distribution][1][1],
        ))
        results.append({})
        for model_name in models:
            results[idx][model_name] = evaluate.test_model(
                models[model_name],
                vectors,
                targets,
                test_vectors,
                test_targets,
                is_classification=False,
            )[0]

    for model_name in models:
        plt.plot(KLs,
            [result[model_name] for result in results],
            label=model_name,
        )

    plt.legend()
    plt.show()

    print('All Done.')
