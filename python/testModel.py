import sys
import os
from os.path import isfile, join, splitext

import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#---------------------------------------------------------------------------
# Globals
#---------------------------------------------------------------------------
#
CATEGORY_NUM = {
    '10s_female' : 0,
    '20s_female' : 1,
    '30s_female' : 2,
    '10s_male'   : 3,
    '20s_male'   : 4,
    '30s_male'   : 5,
}
CATEGORY_NAME = {
    0 : '10s_female',
    1 : '20s_female',
    2 : '30s_female',
    3 : '10s_male',
    4 : '20s_male',
    5 : '30s_male',
}


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def grabAuthors(csv_path, num_files=1e10):

    if num_files <= 0:
        print('You must want at least one file to call grabAuthors.')
        return None

    data_frames = []
    for csv_file_num, csv_file in enumerate(os.listdir(csv_path)):
        if csv_file_num == num_files:
            break
        data_frames.append(pd.read_csv(join(csv_path, csv_file), header=None))

    return pd.concat(data_frames, ignore_index=True)

#
def vectorizeText(texts):
    # Build a vectorizer that splits strings into sequences of 1 to 3 words
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=2,
        use_idf=True,
        sublinear_tf=True,
    )
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=2,
        use_idf=True,
        sublinear_tf=True,
    )
    # Combined Vector
    vectorizer = FeatureUnion([
        ('word:', word_vectorizer),
        ('char:', char_vectorizer),
    ])
    print('Vectorising Data...')
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

#
def reduceDimensionality(vectors, num_dimensions):
    svd = TruncatedSVD(n_components=num_dimensions, random_state=42)
    print(
        'Using Latent Semantic Analysis'
        f'to reduce the dimensions to {num_dimensions}.'
    )
    vectors_reduced = svd.fit_transform(vectors)
    return svd, vectors_reduced


#
def cross_validate_model(model, X_train, y_train):
    # Build a stratified k-fold cross-validator object
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=skf)
    average_score = sum(scores)/len(scores)
    print(f'The cross validation scores are: {scores}')
    print(f'The average cross validation score is: {average_score}')

#
def grabArguments():
    if len(sys.argv) < 2:
        print(
            'Please pass the directory containing the author text CSV files.'
        )
        sys.exit(0)

    csv_path = sys.argv[1]
    return csv_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    NUM_FILES = 2
    NUM_DATAPOINTS = 500

    csv_path = grabArguments()
    data_frame = grabAuthors(csv_path, num_files=2)

    vectorizer, vectors = vectorizeText(data_frame.iloc[:, 3])
    #reducer, vectors_reduced = reduceDimensionality(vectors, 300)

    binary_categories = []
    for idx, item in enumerate(data_frame.iloc[:, 0]):
        binary_categories.append(
            1 if item > 2 else 0
        )

    model = LinearSVC(random_state=42)

    cross_validate_model(model, vectors, binary_categories)
