#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
NUM_DIMENSIONS = 300
TSNE_NUM_DIMENSIONS = 3


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
    if reliability < 0.25:
        return 0
    elif reliability < 0.5:
        return 1
    elif reliability < 0.75:
        return 2
    else:
        return 3

def reduceDimensionality(reduction_technique, vectors, reliability_bins):
    if reduction_technique == 'lsa':
        # Truncated SVD a.k.a. Latent Semantic Analysis
        reducer = TruncatedSVD(n_components=NUM_DIMENSIONS)
        vectors_reduced = reducer.fit_transform(vectors)

    elif reduction_technique == 'tsne':
        reducer = TSNE(n_components=TSNE_NUM_DIMENSIONS)
        vectors_reduced = reducer.fit_transform(vectors)

    elif reduction_technique == 'lda':
        reducer = LinearDiscriminantAnalysis(n_components=NUM_DIMENSIONS)
        vectors_reduced = reducer.fit_transform(
            vectors.toarray(),
            reliability_bins,
        )

    return reducer, vectors_reduced

def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass in order:\n'
            '\tThe vector directory.\n'
            '\tThe reduction technique.\n'
            '\t\tMust be either: \'tsne\',\'lda\', or \'lsa\'.'
        )
        sys.exit(0)

    if sys.argv[2] not in ('tsne', 'lda', 'lsa'):
        print(
            'The reduction technique must be either: '
            '\'tsne\',\'lda\', or \'lsa\'.'
        )
        sys.exit(0)

    vector_dir = sys.argv[1]
    reduction_technique = sys.argv[2]
    return vector_dir, reduction_technique


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    vector_dir, reduction_technique = grabArguments()

    vectors_file = path.join(vector_dir, 'vectors.npz')
    reliabilities_file = path.join(vector_dir, 'reliabilities.npy')

    reducer_file = path.join(
        vector_dir,
        f'{reduction_technique}_reducer.obj',
    )
    reduced_vectors_file = path.join(
        vector_dir,
        f'{reduction_technique}_reduced_vectors',
    )

    print('Loading Vectors...')
    vectors = sparse.load_npz(vectors_file)

    print('Loading Reliabilities...')
    reliabilities = np.load(reliabilities_file, allow_pickle=True)

    print('Grouping Reliabilities into Bins...')
    reliability_bins = [putInBin(item) for item in reliabilities]

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
