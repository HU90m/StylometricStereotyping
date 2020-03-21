#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
USE_THUNDERSVM = True
USE_CATBOOST = True
USE_TENSORFLOW = True
IS_PAN13 = False


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path

from scipy import sparse

import numpy as np


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVC

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


if USE_THUNDERSVM:
    from thundersvm import SVR, SVC
else:
    from sklearn.svm import SVR, SVC

if USE_CATBOOST:
    import catboost as cat

if USE_TENSORFLOW:
    from tensorflow import keras
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


#---------------------------------------------------------------------------
# Globals
#---------------------------------------------------------------------------
#
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
MODELS = (
    'ridge',
    'catboost',
    'nn',
    'lasso',
    'clf_nn',
    'sgd',
    'svr',
    'lin_svc',
    'category_svm',
    'category_catboost',
    'category_nn',
)


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
    jobs=-1,
    splits=10,
):
    if is_classification:
        results = cross_validate(
            model,
            X_train,
            y_train,
            scoring='accuracy',
            return_train_score=calculate_training_scores,
            cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=42),
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
            cv=KFold(n_splits=splits, shuffle=True, random_state=42),
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
):
    dataset = cat.Pool(
        data=X_train,
        label=y_train,
    )
    if is_classification:
        results = cat.cv(
            dataset,
            parameters,
            folds=StratifiedKFold(n_splits=splits, shuffle=True, random_state=42),
        )
    else:
        results = cat.cv(
            dataset,
            parameters,
            folds=KFold(n_splits=splits, shuffle=True, random_state=42),
        )
    print(results)


def make_net(build_num_features=300, build_classifier=False):

    net = keras.models.Sequential()

    # 2 fully connected layers with a ReLU activation function
    net.add(keras.layers.Dense(
        units=16,
        activation='relu',
        input_shape=(build_num_features,),
    ))
    net.add(keras.layers.Dense(
        units=16,
        activation='relu',
        input_shape=(build_num_features,),
    ))

    # fully connected layer with sigmoid activation
    net.add(keras.layers.Dense(
        units=1,
        activation='sigmoid',
    ))

    if build_classifier:
        net.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'],
        )
    else:
        net.compile(
            loss='mean_squared_error',
            optimizer='adam',
        )
    return net


def grabArguments():
    if len(sys.argv) < 4:
        print(
            'Please pass the following in order:\n'
            '\tThe vector file.\n'
            '\tThe target file.\n'
            '\tThe model to be used.\n'
        )
        sys.exit(0)

    if not path.isfile(sys.argv[1]):
        print('The following is not a file:')
        print(f'\t{sys.argv[1]}')
        sys.exit(0)

    if not path.isfile(sys.argv[2]):
        print('The following is not a file:')
        print(f'\t{sys.argv[2]}')
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
    print('Settings:')
    print(f'vectors file:\n\t{vectors_file}')
    print(f'targets file:\n\t{targets_file}')
    print(f'model:\n\t{model}')

    print('Loading Vectors...')
    vectors_filename, vectors_file_ext = path.splitext(vectors_file)
    if vectors_file_ext == '.npz':
        vectors = sparse.load_npz(vectors_file)
        is_sparse = True
    else:
        vectors = np.load(vectors_file, allow_pickle=True)
        is_sparse = False


    print('Loading Reliabilities...')
    targets = np.load(targets_file, allow_pickle=True)


    if model == 'ridge':
        print('Cross Validating Model...')
        cross_validate_model(Ridge(), vectors, targets)

    elif model == 'catboost':
        if not USE_CATBOOST:
            print('catboost needs to be enabled to use this model')
        else:
            print('Cross Validating Model...')
            parameters = {
                "iterations": 2,
                "depth": 2,
                "verbose": False,
                "task_type": "GPU",
            }
            cross_validate_cat_model(
                parameters,
                vectors,
                targets,
                is_classification=False,
            )

    elif model == 'nn':
        if not USE_TENSORFLOW:
            print('tensorflow needs to be enabled to use this model')
        elif is_sparse:
            print('This model does not support sparse matrices at present.')
        else:
            print('Building Model...')
            network = KerasRegressor(
                build_fn=make_net,
                build_num_features=vectors.shape[1],
                build_classifier=False,
                epochs=1,
                batch_size=100,
                verbose=0,
            )
            print('Cross Validating Model...')
            cross_validate_model(
                network,
                vectors,
                targets,
                jobs=None,
                splits=2,
            )

    elif model == 'lasso':
        print('Cross Validating Model...')
        cross_validate_model(Lasso(), vectors, targets)

    elif model == 'sgd':
        print('Cross Validating Model...')
        cross_validate_model(SGDRegressor(), vectors, targets)

    elif model == 'svr':
        print('Cross Validating Model...')
        jobs = None if USE_THUNDERSVM else -1
        cross_validate_model(SVR(), vectors, targets, jobs=jobs)

    elif model == 'lin_svc':
        print('Grouping Reliabilities into Bins...')
        reliability_bins = [putIn2Bins(item) for item in targets]

        print('Cross Validating Model...')
        cross_validate_model(
            LinearSVC(),
            vectors,
            reliability_bins,
            is_classification=True,
        )

    elif model == 'category_svm':
        if IS_PAN13:
            print('Grouping Data into Female/Male Bins...')
            targets = [MaleFemaleSplit(item) for item in targets]

        print('Cross Validating Model...')
        jobs = None if USE_THUNDERSVM else -1
        cross_validate_model(
            SVC(),
            vectors,
            targets,
            is_classification=True,
            jobs=jobs,
            splits=2,
        )

    elif model == 'category_catboost':
        if not USE_CATBOOST:
            print('catboost needs to be enabled to use this model')
        else:
            if IS_PAN13:
                print('Grouping Data into Female/Male Bins...')
                targets = [MaleFemaleSplit(item) for item in targets]

            print('Cross Validating Model...')
            parameters = {
                "iterations": 2,
                "depth": 2,
                "verbose": False,
                "task_type": "GPU",
            }
            cross_validate_cat_model(
                parameters,
                vectors,
                targets,
                is_classification=True,
                splits=3,
            )

    elif model == 'category_nn':
        if not USE_TENSORFLOW:
            print('tensorflow needs to be enabled to use this model')
        elif is_sparse:
            print('This model does not support sparse matrices at present.')
        else:
            if IS_PAN13:
                print('Grouping Data into Female/Male Bins...')
                targets = [MaleFemaleSplit(item) for item in targets]

            print('Building Model...')
            network = KerasClassifier(
                build_fn=make_net,
                build_num_features=vectors.shape[1],
                build_classifier=False,
                epochs=1,
                batch_size=100,
                verbose=0,
            )

            print('Cross Validating Model...')
            cross_validate_model(
                network,
                vectors,
                targets,
                is_classification=True,
                jobs=None,
                splits=2,
            )

    print('All Done.')
