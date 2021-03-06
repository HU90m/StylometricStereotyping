#!/bin/python3.8
#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
IMPORT_REDUCER = False
LSA_NUM_DIMENSIONS = 300
LDA_NUM_DIMENSIONS = None
TSNE_NUM_DIMENSIONS = 3
LDA_FIT_PERCENTAGE = 0.25


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path
import pickle

import numpy as np
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def putInBin(reliability):
    if reliability < 0.125:
        return 0
    elif reliability < 0.25:
        return 1
    elif reliability < 0.375:
        return 2
    elif reliability < 0.5:
        return 3
    elif reliability < 0.625:
        return 4
    elif reliability < 0.75:
        return 2
    elif reliability < 0.875:
        return 2
    else:
        return 3

def reduceDimensionality(reduction_technique, vectors, reliability_bins):
    if reduction_technique == 'lsa':
        # Truncated SVD a.k.a. Latent Semantic Analysis
        reducer = TruncatedSVD(n_components=LSA_NUM_DIMENSIONS)
        vectors_reduced = reducer.fit_transform(vectors)

    elif reduction_technique == 'tsne':
        reducer = TSNE(
            n_components=TSNE_NUM_DIMENSIONS,
            n_jobs=-1,
        )
        vectors_reduced = reducer.fit_transform(vectors)

    elif reduction_technique == 'lda':
        fit_index_max = int(vectors.shape[0] * LDA_FIT_PERCENTAGE)
        print(f'fitting up to index {fit_index_max} out of {vectors.shape[0]}')
        reducer = LinearDiscriminantAnalysis(n_components=LDA_NUM_DIMENSIONS)
        reducer.fit(
            vectors.toarray()[:fit_index_max],
            reliability_bins[:fit_index_max],
        )
        vectors_reduced = reducer.transform(vectors.toarray())

    return reducer, vectors_reduced

def grabArguments():
    if len(sys.argv) < 6:
        print(
            'Please pass in order:\n'
            '\tThe input vector file.\n'
            '\tThe reliabilities file.\n'
            '\tThe output (reduced) vector file.\n'
            '\tThe reducer file.\n'
            '\tThe reduction technique.\n'
            '\t\tMust be either: \'tsne\',\'lda\', or \'lsa\'.'
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

    if sys.argv[5] not in ('tsne', 'lda', 'lsa'):
        print(
            'The reduction technique must be either: '
            '\'tsne\',\'lda\', or \'lsa\'.'
        )
        sys.exit(0)

    vectors_file = sys.argv[1]
    reliabilities_file = sys.argv[2]
    reduced_vectors_file = sys.argv[3]
    reducer_file = sys.argv[4]
    reduction_technique = sys.argv[5]
    return (
        vectors_file,
        reliabilities_file,
        reduced_vectors_file,
        reducer_file,
        reduction_technique,
    )


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    (
        vectors_file,
        reliabilities_file,
        reduced_vectors_file,
        reducer_file,
        reduction_technique,
    ) = grabArguments()


    print('Loading Vectors...')
    vectors = sparse.load_npz(vectors_file)

    print('Loading Reliabilities...')
    reliabilities = np.load(reliabilities_file, allow_pickle=True)

    print('Grouping Reliabilities into Bins...')
    reliability_bins = [putInBin(item) for item in reliabilities]

    if IMPORT_REDUCER:
        print('Reducing Data...')
        with open(reducer_file, 'rb') as reducer_file_p:
            reducer = pickle.load(reducer_file_p)
        if reduction_technique == 'lda':
            reduced_vectors = reducer.transform(vectors.toarray())
        else:
            reduced_vectors = reducer.transform(vectors)
    else:
        print('Reducing Data...')
        reducer, reduced_vectors = reduceDimensionality(
            reduction_technique,
            vectors,
            reliability_bins,
        )

        print('Saving Reducer...')
        with open(reducer_file, 'wb') as reducer_file_p:
            pickle.dump(reducer, reducer_file_p)


    print('Saving Reduced Vectors...')
    np.save(reduced_vectors_file, reduced_vectors)

    print('All Done.')
