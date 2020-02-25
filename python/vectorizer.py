import sys
import os
from os.path import isfile, join, splitext

import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion



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
        min_df=0.01,# ignore terms in only 1% of documents
        use_idf=True,
        sublinear_tf=True,
    )
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=0.01,# ignore terms in only 1% of documents
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
def grabArguments():
    if len(sys.argv) < 2:
        print(
            'Please pass in order:\n'
            '\tThe directory containing the author text CSV files.\n'
            '\tThe name of output vector file.\n'
            '\tThe name of output categories file.'
        )
        sys.exit(0)

    csv_path = sys.argv[1]
    vectors_file = sys.argv[2]
    categories_file = sys.argv[3]
    return csv_path, vectors_file, categories_file


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':

    csv_path, vectors_file, categories_file = grabArguments()
    data_frame = grabAuthors(csv_path, num_files=1)

    data_frame = data_frame.iloc[0:100,:]

    print(data_frame)

    vectorizer, vectors = vectorizeText(data_frame.iloc[:, 3])

    #print('Fitting Data...')
    #vectors = vectorizer.transform(data_frame.iloc[:, 3])

    #print('Reducing Dimensionality...')
    #reducer, vectors_reduced = reduceDimensionality(vectors, 300)

    print('Saving Vectors...')
    sparse.save_npz(vectors_file, vectors)

    print('Saving Categories...')
    np.save(categories_file, np.array(data_frame.iloc[:, 0]))

    print('All Done')
