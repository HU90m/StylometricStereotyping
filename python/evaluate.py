#!/bin/python3
#---------------------------------------------------------------------------
# Constants
#---------------------------------------------------------------------------
#
IS_PAN13 = False
if IS_PAN13:
    CATEGORY_NUM = {
        '30s_female' : 0,
        '30s_male'   : 1,
        '20s_male'   : 2,
        '20s_female' : 3,
        '10s_female' : 4,
        '10s_male'   : 5,
    }
else:
    CATEGORY_NUM = {
        'bot'   : 0,
        'human' : 1,
    }
OPERATIONS = (
    'param',
    'cv',
    'test',
)
MODELS = (
    'ridge',
    'lasso',
    'sgd',
    'lin_svr',
    'lin_svc',
    'svr',
    'svc',
    'catboost',
    'catboost_c',
)


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path

import numpy as np
import catboost as cat

from scipy import sparse

from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.svm import SVR, SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.utils.fixes import loguniform

#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def PAN13MaleFemaleSplit(category):
    if (category == CATEGORY_NUM['30s_female']
        or category ==  CATEGORY_NUM['20s_female']):
        return 0
    else:
        return 1

def param_opt_model(
    model,
    params,
    vectors,
    targets,
    is_classification=False,
    calculate_training_scores=True,
    grid_search=True,
    seed=42,
    iterations=20,
    jobs=-1,
    splits=5,
):
    print('Searching Parameter Space...')
    if is_classification:
        if grid_search:
            param_finder = GridSearchCV(
                model,
                params,
                scoring='accuracy',
                return_train_score=calculate_training_scores,
                cv=StratifiedKFold(
                    n_splits=splits,
                    shuffle=True,
                    random_state=seed,
                ),
                n_jobs=jobs,
            )
        else:
            param_finder = RandomizedSearchCV(
                model,
                params,
                scoring='accuracy',
                return_train_score=calculate_training_scores,
                cv=StratifiedKFold(
                    n_splits=splits,
                    shuffle=True,
                    random_state=seed,
                ),
                n_jobs=jobs,
                n_iter=iterations,
            )
    else:
        if grid_search:
            param_finder = GridSearchCV(
                model,
                params,
                scoring='r2',
                return_train_score=calculate_training_scores,
                cv=KFold(
                    n_splits=splits,
                    shuffle=True,
                    random_state=seed,
                ),
                n_jobs=jobs,
            )
        else:
            param_finder = RandomizedSearchCV(
                model,
                params,
                scoring='r2',
                return_train_score=calculate_training_scores,
                cv=KFold(
                    n_splits=splits,
                    shuffle=True,
                    random_state=seed,
                ),
                n_jobs=jobs,
                n_iter=iterations,
            )

    results = param_finder.fit(vectors, targets)
    print(f'best parameters found: {results.best_params_}')
    print(f'score achieved by these parameters: {results.best_score_}')
    print(f'best parameters found:')
    for key, item in results.best_estimator_.get_params().items():
        key += ' ' * (15 - len(key))
        print(f'\t{key}\t{item}')

def cross_validate_model(
    model,
    X_train,
    y_train,
    is_classification=False,
    calculate_training_scores=True,
    jobs=-1,
    splits=10,
    seed=42,
):
    print('Cross Validating Model...')
    if is_classification:
        results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='accuracy',
            return_train_score=calculate_training_scores,
            cv=StratifiedKFold(
                n_splits=splits,
                shuffle=True,
                random_state=seed,
            ),
            n_jobs=jobs,
        )
        display = [
            ('fit time', 'fit_time'),
            ('test accuracy', 'test_score'),
        ]
        if calculate_training_scores:
            display.append(
                ('training accuracy', 'train_score')
            )
    else:
        results = cross_validate(
            model,
            X_train,
            y_train,
            scoring=(
                'neg_root_mean_squared_error',
                'r2',
            ),
            return_train_score=calculate_training_scores,
            cv=KFold(n_splits=splits, shuffle=True, random_state=seed),
            n_jobs=jobs,
        )
        display = [
            ('fit time', 'fit_time'),
            (
                'test root mean squared error',
                'test_neg_root_mean_squared_error',
            ),
            (
                'test R2',
                'test_r2',
            ),
        ]
        if calculate_training_scores:
            display.append((
                'training root mean squared error',
                'train_neg_root_mean_squared_error',
            ))
            display.append((
                'training R2',
                'train_r2',
            ))

    for name, key in display:
        print(f'The {name}s:')
        for metric in results[key]:
            print(f'\t{metric}')
        average_metric = sum(results[key])/len(results[key])
        print(f'The average {name}:')
        print(f'\t{average_metric}\n')

def cross_validate_cat_model(
    model_parameters,
    X_train,
    y_train,
    is_classification=False,
    splits=10,
    seed=42,
):
    print('Cross Validating Model...')
    dataset = cat.Pool(
        data=X_train,
        label=y_train,
    )
    if is_classification:
        results = cat.cv(
            dataset,
            parameters,
            folds=StratifiedKFold(
                n_splits=splits,
                shuffle=True,
                random_state=seed,
            ),
        )
    else:
        results = cat.cv(
            dataset,
            parameters,
            folds=KFold(
                n_splits=splits,
                shuffle=True,
                random_state=seed,
            ),
        )
    print(results)

def test_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    is_classification=False,
):
    print('Testing Model...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        accuracy_result = accuracy_score(y_test, y_pred)
        print(f'The models accuracy is {accuracy_result}')
        return accuracy_result
    else:
        r2_result = r2_score(y_test, y_pred)
        print(f'The models r2 score is {r2_result}')
        rmse_result = mean_squared_error(y_test, y_pred, squared=False)
        print(f'The models root mean squared error is {rmse_result}')
        mae_result = mean_absolute_error(y_test, y_pred)
        print(f'The models mean absolute error is {mae_result}')
        return (
            r2_result,
            rmse_result,
            mae_result,
        )

def grabArguments():
    if len(sys.argv) < 5:
        print(
            'Please pass the following in order:\n'
            '\tThe operation.\n'
            '\tThe model to be used.\n'
            '\tThe vector file.\n'
            '\tThe target file.\n'
            '\t[test] The test vector file.\n'
            '\t[test] The test target file.\n'
        )
        sys.exit(0)

    if sys.argv[1] not in OPERATIONS:
        print('The available operations are:')
        for operation in OPERATIONS:
            print(f'\t{operation}')
        sys.exit(0)

    if sys.argv[2] not in MODELS:
        print('The available models are:')
        for model in MODELS:
            print(f'\t{model}')
        sys.exit(0)

    if not path.isfile(sys.argv[3]):
        print('The following is not a file:')
        print(f'\t{sys.argv[2]}')
        sys.exit(0)

    if not path.isfile(sys.argv[4]):
        print('The following is not a file:')
        print(f'\t{sys.argv[3]}')
        sys.exit(0)

    operation = sys.argv[1]
    model = sys.argv[2]
    vectors_file = sys.argv[3]
    targets_file = sys.argv[4]

    if operation == 'test':
        if len(sys.argv) < 7:
            print(
                'Please pass the following in order:\n'
                '\tThe operation.\n'
                '\tThe vector file.\n'
                '\tThe target file.\n'
                '\tThe model to be used.\n'
                '\t[test] The test vector file.\n'
                '\t[test] The test target file.\n'
            )
            sys.exit(0)

        if not path.isfile(sys.argv[5]):
            print('The following is not a file:')
            print(f'\t{sys.argv[6]}')
            sys.exit(0)

        if not path.isfile(sys.argv[6]):
            print('The following is not a file:')
            print(f'\t{sys.argv[6]}')
            sys.exit(0)

        test_vectors_file = sys.argv[5]
        test_targets_file = sys.argv[6]
    else:
        test_vectors_file = None
        test_targets_file = None

    return (
        operation,
        vectors_file,
        targets_file,
        model,
        test_vectors_file,
        test_targets_file,
    )


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    (
        operation,
        vectors_file,
        targets_file,
        model,
        test_vectors_file,
        test_targets_file,
    ) = grabArguments()
    OP_TEST = operation == 'test'
    OP_CV = operation == 'cv'
    OP_PARAM = operation == 'param'

    print('Settings:')
    print(f'operation:\n\t{operation}')
    print(f'vectors file:\n\t{vectors_file}')
    print(f'targets file:\n\t{targets_file}')
    print(f'model:\n\t{model}')
    if OP_TEST:
        print(f'test vectors file:\n\t{test_vectors_file}')
        print(f'test targets file:\n\t{test_targets_file}')


    print('Loading Vectors...')
    vectors_filename, vectors_file_ext = path.splitext(vectors_file)
    if vectors_file_ext == '.npz':
        vectors = sparse.load_npz(vectors_file)
        is_sparse = True
    else:
        vectors = np.load(vectors_file, allow_pickle=True)
        is_sparse = False
    print(f'loaded vectors file of shape: {vectors.shape}')

    print('Loading Targets...')
    targets = np.load(targets_file, allow_pickle=True)
    print(f'loaded target file of shape: {targets.shape}')

    if OP_TEST:
        print('Loading Test Vectors...')
        (
            test_vectors_filename,
            test_vectors_file_ext,
        ) = path.splitext(test_vectors_file)

        if test_vectors_file_ext == '.npz':
            test_vectors = sparse.load_npz(test_vectors_file)
            is_sparse = True
        else:
            test_vectors = np.load(test_vectors_file, allow_pickle=True)
            is_sparse = False
        print(f'loaded test vectors file of shape: {test_vectors.shape}')

        print('Loading Test Targets...')
        test_targets = np.load(test_targets_file, allow_pickle=True)
        print(f'loaded test target file of shape: {test_targets.shape}')


    if IS_PAN13 and model in ('lin_svc', 'svc', 'catboost_c'):
        print('Grouping Data into Female/Male Bins...')
        targets = [PAN13MaleFemaleSplit(item) for item in targets]
        if OP_TEST:
            test_targets = [PAN13MaleFemaleSplit(item) for item in test_targets]


    if model == 'ridge':
        if OP_CV:
            cross_validate_model(Ridge(), vectors, targets)
        elif OP_TEST:
            test_model(Ridge(), vectors, targets, test_vectors, test_targets)
        elif OP_PARAM:
            params = {
                'tol': loguniform(1e-6, 1e-2),
                #'tol': (1e-6, 1e-5, 1e-4, 1e-3),
            }
            param_opt_model(Ridge(), params, vectors, targets, grid_search=False, iterations=100, seed=None)

    elif model == 'lasso':
        if OP_CV:
            cross_validate_model(Lasso(), vectors, targets)
        elif OP_TEST:
            test_model(Lasso(), vectors, targets, test_vectors, test_targets)

    elif model == 'sgd':
        if OP_CV:
            cross_validate_model(SGDRegressor(), vectors, targets)
        elif OP_TEST:
            test_model(SGDRegressor(), vectors, targets, test_vectors, test_targets)

    elif model == 'svr':
        if OP_CV:
            cross_validate_model(SVR(), vectors, targets)
        elif OP_TEST:
            test_model(SVR(), vectors, targets, test_vectors, test_targets)
        elif OP_PARAM:
            params = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
            param_opt_model(SVR(), params, vectors, targets)

    elif model == 'lin_svr':
        if OP_CV:
            cross_validate_model(LinearSVR(), vectors, targets, splits=3)
        elif OP_TEST:
            test_model(
                LinearSVR(),
                vectors,
                targets,
                test_vectors,
                test_targets,
            )

    elif model == 'svc':
        if OP_CV:
            cross_validate_model(
                SVC(),
                vectors,
                targets,
                is_classification=True,
                splits=3,
            )
        elif OP_TEST:
            test_model(
                SVC(),
                vectors,
                targets,
                test_vectors,
                test_targets,
                is_classification=True,
            )

    elif model == 'lin_svc':
        if OP_CV:
            cross_validate_model(
                LinearSVC(),
                vectors,
                targets,
                is_classification=True,
                splits=4,
            )
        elif OP_TEST:
            test_model(
                LinearSVC(),
                vectors,
                targets,
                test_vectors,
                test_targets,
                is_classification=True,
            )

    elif model == 'catboost':
        parameters = {
            "iterations": 100,
            "depth": 6,
            "verbose": False,
            "task_type": "GPU",
        }
        if OP_CV:
            cross_validate_model(
                cat.CatBoostRegressor(**parameters),
                vectors,
                targets,
                splits=2,
                jobs=None,
            )
        elif OP_TEST:
            test_model(
                cat.CatBoostRegressor(**parameters),
                vectors,
                targets,
                test_vectors,
                test_targets,
                is_classification=False,
            )

    elif model == 'catboost_c':
        parameters = {
            "iterations": 2,
            #"depth": 2,
            "verbose": False,
            "task_type": "GPU",
        }
        if OP_CV:
            cross_validate_model(
                cat.CatBoostClassifier(**parameters),
                vectors,
                targets,
                splits=2,
                jobs=None,
            )
        elif OP_TEST:
            test_model(
                cat.CatBoostClassifier(**parameters),
                vectors,
                targets,
                test_vectors,
                test_targets,
                is_classification=True,
            )

    print('All Done.')
