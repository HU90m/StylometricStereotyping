import sys
import os
from os import path
import pickle

import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

PROGRESS_BAR=True
if PROGRESS_BAR:
    from tqdm import tqdm

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
    if PROGRESS_BAR:
        vectors = vectorizer.fit_transform(tqdm(texts))
    else:
        vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass in order:\n'
            '\tThe directory containing the author text CSV files.\n'
            '\tThe name of output path.'
        )
        sys.exit(0)

    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    return csv_path, output_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    csv_path, output_path = grabArguments()

    vectorizer_file = path.join(output_path, 'vectorizer.obj')
    vectors_file = path.join(output_path, 'vectors.npz')
    categories_file = path.join(output_path, 'categories')


    print('Importing CSV...')
    data_frame = grabAuthors(csv_path)

    print('Vectorising Data...')
    vectorizer, vectors = vectorizeText(data_frame.iloc[:, 3])

    print('Saving Vectorizer...')
    with open(vectorizer_file, 'wb') as vec_file_p:
        pickle.dump(vectorizer, vec_file_p)

    print('Saving Vectors...')
    sparse.save_npz(vectors_file, vectors)

    print('Saving Categories...')
    np.save(categories_file, np.array(data_frame.iloc[:, 0]))

    print('All Done')
