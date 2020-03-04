#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
CLASSIFICATION = False


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path

from scipy import sparse

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def cross_validate_model(model, X_train, y_train):
    # Build a stratified k-fold cross-validator object
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=skf)
    average_score = sum(scores)/len(scores)
    print(f'The cross validation scores are:')
    for score in scores:
        print(f'\t{score}')
    print(f'The average cross validation score is:')
    print(f'\t{average_score}')

def grabArguments():
    if len(sys.argv) < 2:
        print('Please pass the vector directory.')
        sys.exit(0)

    vector_dir = sys.argv[1]
    return vector_dir


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    vector_dir = grabArguments()

    vectors_file = path.join(vector_dir, 'vectors.npz')
    categories_file = path.join(vector_dir, 'categories.npy')

    print('Loading Vectors...')
    vectors = sparse.load_npz(vectors_file)

    print('Loading Categories...')
    categories = np.load(categories_file, allow_pickle=True)

    print('Selecting Categories...')
    # selects only categories 2 and 5
    selection = np.logical_or.reduce([categories == x for x in (2, 5)])

    selected_vectors = vectors.todense()[selection]
    selected_categories = categories[selection]

    binary_selected_categories = \
        [0 if item==2 else 1 for item in selected_categories]

    print('Cross Validating Model...')
    #model = LinearSVC(random_state=42)
    model = SGDClassifier(random_state=42)

    cross_validate_model(model, selected_vectors, binary_selected_categories)

    print('All Done')
