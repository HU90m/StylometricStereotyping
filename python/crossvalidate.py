#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
MODELS = ('ridge', 'svr', 'lin_svc')


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path

from scipy import sparse

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.svm import LinearSVC

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def putIn2Bins(reliability):
    if reliability < 0.5:
        return 0
    else:
        return 1

def putIn4Bins(reliability):
    if reliability < 0.25:
        return 0
    elif reliability < 0.5:
        return 1
    elif reliability < 0.75:
        return 2
    else:
        return 3

def cross_validate_model(
    model,
    X_train,
    y_train,
    is_classification=False,
    calculate_training_scores=True,
):

    if is_classification:
        results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='accuracy',
            return_train_score=calculate_training_scores,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=-1,
        )
    else:
        results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='neg_mean_squared_error',
            return_train_score=calculate_training_scores,
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=-1,
        )

    display = [
        ('fit time', 'fit_time'),
        ('test score', 'test_score'),
    ]
    if calculate_training_scores:
        display.append(
            ('training score', 'train_score')
        )

    for name, key in display:
        print(f'The {name}s:')
        for metric in results[key]:
            print(f'\t{metric}')
        average_metric = sum(results[key])/len(results[key])
        print(f'The average {name}:')
        print(f'\t{average_metric}\n')


def grabArguments():
    if len(sys.argv) < 4:
        print(
            'Please pass the following in order:\n'
            '\tThe vector file.\n'
            '\tThe target file.\n'
            '\tThe model to be used.\n'
        )
        sys.exit(0)

    if sys.argv[3] not in MODELS:
        print('The available models are:')
        for model in MODELS:
            print(f'\t{model}')
        sys.exit(0)

    vectors_file = sys.argv[1]
    targets_file = sys.argv[2]
    model = sys.argv[3]
    return vectors_file, targets_file, model


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    vectors_file, targets_file, model = grabArguments()

    print('Loading Vectors...')
    vectors_filename, vectors_file_ext = path.splitext(vectors_file)
    if vectors_file_ext == '.npz':
        vectors = sparse.load_npz(vectors_file)
        sparse = True
    else:
        vectors = np.load(vectors_file, allow_pickle=True)
        sparse = False

    print('Loading Reliabilities...')
    reliabilities = np.load(targets_file, allow_pickle=True)

    if model in ('lin_svc',):
        print('Grouping Reliabilities into Bins...')
        reliability_bins = [putIn2Bins(item) for item in reliabilities]

    print('Cross Validating Model...')
    if model == 'ridge':
        cross_validate_model(Ridge(), vectors, reliabilities)

    elif model == 'svr':
        cross_validate_model(SVR(), vectors, reliabilities)

    elif model == 'lin_svc':
        cross_validate_model(
            LinearSVC(),
            vectors,
            reliability_bins,
            is_classification=True,
        )

    print('All Done.')
