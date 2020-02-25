import sys
from os import path
import pickle

import numpy as np
from scipy import sparse

from sklearn.decomposition import TruncatedSVD


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def reduceDimensionality(vectors, num_dimensions):
    svd = TruncatedSVD(n_components=num_dimensions, random_state=42)

    # using Latent Semantic Analysis to reduce the dimensions
    vectors_reduced = svd.fit_transform(vectors)
    return svd, vectors_reduced

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

    NUM_DIMENSIONS = 30

    vectors_file = path.join(vector_dir, 'vectors.npz')
    reducer_file = path.join(vector_dir, 'reducer.obj')
    reduced_vectors_file = path.join(vector_dir, 'reduced_vectors')

    print('Loading Vectors...')
    vectors = sparse.load_npz(vectors_file)

    print('Reducing Data...')
    reducer, reduced_vectors = reduceDimensionality(vectors, NUM_DIMENSIONS)

    print('Saving Reducer...')
    with open(reducer_file, 'wb') as reducer_file_p:
        pickle.dump(reducer, reducer_file_p)

    print('Saving Reduced Vectors...')
    np.save(reduced_vectors_file, reduced_vectors)

    print('All Done')
