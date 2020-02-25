import sys

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
    print(f'The cross validation scores are: {scores}')
    print(f'The average cross validation score is: {average_score}')

def grabArguments():
    if len(sys.argv) < 2:
        print(
            'Please pass:\n'
            '\tThe name of input vector file.\n'
            '\tThe name of input categories file.'
        )
        sys.exit(0)

    vectors_file = sys.argv[1]
    categories_file = sys.argv[2]
    return vectors_file, categories_file

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':

    vectors_file, categories_file = grabArguments()

    vectors = sparse.load_npz(vectors_file)

    categories = np.load(categories_file, allow_pickle=True)

    # selects only categories 2 and 5
    selection = np.logical_or.reduce([categories == x for x in (2, 5)])

    selected_vectors = vectors.todense()[selection]
    selected_categories = categories[selection]

    print(selected_categories)

    selected_categories = [0 if item==2 else 1 for item in selected_categories]

    print(selected_categories)


    #model = LinearSVC(random_state=42)
    model = SGDClassifier(random_state=42)

    cross_validate_model(model, selected_vectors, selected_categories)
