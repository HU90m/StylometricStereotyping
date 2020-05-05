#!/bin/python3
#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
import os
from os import path
import pickle

import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def grabAuthors(csv_path, num_files=-1):
    data_frames = []
    for csv_file_num, csv_file in enumerate(os.listdir(csv_path)):
        if csv_file_num == num_files:
            break
        data_frames.append(pd.read_csv(path.join(csv_path, csv_file), header=None))

    return pd.concat(data_frames, ignore_index=True)

def vectorizeText(texts):
    # Build a vectorizer that splits strings into sequences of 1 to 3 words
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0.01,# ignore terms in only 1% of documents
        max_df=1.0,# ignore terms in 100% of documents
        use_idf=True,
        sublinear_tf=True,
    )
    # Build a vectorizer that splits strings into sequences of 3 to 5 characters
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=0.01,# ignore terms in only 1% of documents
        max_df=1.0,# ignore terms in 100% of documents
        use_idf=True,
        sublinear_tf=True,
    )
    # Combined Vector
    vectorizer = FeatureUnion([
        ('word:', word_vectorizer),
        ('char:', char_vectorizer),
    ])
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass in order:\n'
            '\tThe directory containing the author text CSV files.\n'
            '\tThe name of output path.\n'
            '\t[optional] The directory containing the test CSV files.\n'
        )
        sys.exit(0)

    if not path.isdir(sys.argv[1]):
        print(f'The following is not a directory:\n\t{sys.argv[1]}')
        sys.exit(0)
    csv_path = sys.argv[1]

    if not path.isdir(sys.argv[2]):
        print(f'The following is not a directory:\n\t{sys.argv[2]}')
        sys.exit(0)
    output_path = sys.argv[2]

    if len(sys.argv) > 3:
        if not path.isdir(sys.argv[3]):
            print(f'The following is not a directory:\n\t{sys.argv[3]}')
            sys.exit(0)
        test_csv_path = sys.argv[3]
    else:
        test_csv_path = None

    return csv_path, output_path, test_csv_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    csv_path, output_path, test_csv_path = grabArguments()

    vectorizer_file = path.join(output_path, 'vectorizer.obj')
    vectors_file = path.join(output_path, 'vectors.npz')
    categories_file = path.join(output_path, 'categories')


    print('Importing CSV...')
    data_frame = grabAuthors(csv_path)

    print('Vectorising Data...')
    vectorizer, vectors = vectorizeText(data_frame.iloc[:, 3])

    print('Saving Vectoriser...')
    with open(vectorizer_file, 'wb') as vec_file_p:
        pickle.dump(vectorizer, vec_file_p)

    print('Saving Vectors...')
    sparse.save_npz(vectors_file, vectors)

    print('Saving Categories...')
    np.save(categories_file, np.array(data_frame.iloc[:, 0]))

    if test_csv_path:
        test_vectors_file = path.join(output_path, 'test-vectors.npz')
        test_categories_file = path.join(output_path, 'test-categories')

        print('Importing Test CSV...')
        test_data_frame = grabAuthors(test_csv_path)

        print('Vectorising Test Data...')
        test_vectors = vectorizer.transform(test_data_frame.iloc[:, 3])

        print('Saving Test Vectors...')
        sparse.save_npz(test_vectors_file, test_vectors)

        print('Saving Test Categories...')
        np.save(test_categories_file, np.array(test_data_frame.iloc[:, 0]))

    print('All Done')
