#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path

from scipy import sparse

import numpy as np

#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass the following in order:\n'
            '\tThe vector or matrix file.\n'
            '\tThe index past which data should be ignored.\n'
        )
        sys.exit(0)

    if not path.isfile(sys.argv[1]):
        print('The following is not a file:')
        print(f'\t{sys.argv[1]}')
        sys.exit(0)

    data_file = sys.argv[1]
    until_idx = int(sys.argv[2])
    return data_file, until_idx


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    data_file, until_idx = grabArguments()

    print('Loading Data...')
    data_filename, data_file_ext = path.splitext(data_file)
    if data_file_ext == '.npz':
        data = sparse.load_npz(data_file)
        is_sparse = True
    else:
        data = np.load(data_file, allow_pickle=True)
        is_sparse = False
    print(f'input shape: {data.shape}')

    print('Truncating Data...')
    if is_sparse:
        data = data.tocsr()[:until_idx]
    else:
        data = data[:until_idx]
    print(f'output shape: {data.shape}')


    print('Saving Data...')
    output_file = data_filename + '_truncated'

    if is_sparse:
        output_file += data_file_ext
        sparse.save_npz(output_file, data)
    else:
        np.save(output_file, data)
