import sys
from os import listdir
from os.path import isfile, join, splitext

from xml.etree import ElementTree
from html.parser import HTMLParser

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#---------------------------------------------------------------------------
# Global Variables
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
FILENAMES_FILE = '../en_filenames.txt'
NUM_AUTHORS = 500


#---------------------------------------------------------------------------
# Classes
#---------------------------------------------------------------------------
#
class HTML2Text(HTMLParser):
    """Removes HTML tags from strings
    """
    _text = ''

    def handle_endtag(self, tag):
        #if tag == 'br':
        #    self.text += '\n'
        #else:
        #    self.text += ' '
        self._text += ' '

    def handle_data(self, data):
        self._text += data

    @property
    def text(self):
        return self._text


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def grabAuthorText(author_file_path):
    html2text = HTML2Text()

    tree = ElementTree.parse(
        author_file_path,
        parser=ElementTree.XMLParser(encoding="utf-8"),
    )
    root = tree.getroot()
    conversations = root[0]

    for conversation in conversations:
        html2text.feed(conversation.text)

    return html2text.text.lower()

#
def grabAuthors(num_authors):
    authors = {
        'id'       : [],
        'category' : [],
        'filename' : [],
        'text'     : [],
        'len_text' : [],
    }

    author_filenames = open(join(DATAPATH, FILENAMES_FILE))

    print('Getting Authors...')
    for file_num, author_file in enumerate(author_filenames):
        if file_num > num_authors:
            break
        if not file_num % 1000:
            print(f'getting authors {file_num} to {file_num +999}')

        author_file = author_file[:-1]
        author_file_path = join(DATAPATH, author_file)
        file_no_ext, file_ext = splitext(author_file)

        info_from_filename = file_no_ext.split('_', 2)

        if isfile(author_file_path) and (file_ext == '.xml'):
            try:
                author_text = grabAuthorText(author_file_path)

                authors['id'].append(int(info_from_filename[0], 16))
                authors['category'].append(CATEGORY_NUM[info_from_filename[2]])
                authors['filename'].append(author_file)
                authors['text'].append(author_text)
                authors['len_text'].append(len(author_text))

            except TypeError:
                print(f'Type error when Author with id {info_from_filename[0]} '
                    'was being processed.')
                print('Skipping this author.')
        else:
            print(f'{author_file_path} is not an xml file!')
    author_filenames.close()
    return authors

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


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':

    DATAPATH = '/home/hugo/ln/ip/data/training/en'
    if len(sys.argv) < 2:
        print('Please pass the data path.')
        sys.exit(0)

    DATAPATH = sys.argv[1]

    authors = grabAuthors(100)

    #vectorizer, vectors = vectorizeText(authors['text'])
    #reducer, vectors_reduced = reduceDimensionality(vectors, 300)

    binary_categories = []
    for idx, item in enumerate(authors['category']):
        binary_categories.append(
            1 if item > 2 else 0
        )

    model = LinearSVC(random_state=42)

    #cross_validate_model(model, vectors, binary_categories)
