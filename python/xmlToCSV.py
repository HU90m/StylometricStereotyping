import sys
import os
from os.path import isfile, join, splitext

from multiprocessing import Pool, Lock

import csv

from xml.etree import ElementTree
from html.parser import HTMLParser


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
BATCH_SIZE = 10000
NUM_AUTHORS = 1e20
NUM_PROCESSES = 8


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
def createCSVFile(
    xml_file_path,
    xml_file_names,
    csv_file_path,
    csv_file_name,
    print_lock
):
    with print_lock:
        print(f'Making \'{csv_file_name}\'.')

    csv_file_location = join(csv_file_path, csv_file_name)
    csv_file = open(csv_file_location, 'w')
    csv_writer = csv.writer(
        csv_file,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )
    for file_num, author_file in enumerate(xml_file_names):

        author_file_path = join(xml_file_path, author_file)
        file_no_ext, file_ext = splitext(author_file)

        info_from_filename = file_no_ext.split('_', 2)

        if isfile(author_file_path) and (file_ext == '.xml'):
            try:
                author_text = grabAuthorText(author_file_path)
                csv_writer.writerow([
                    CATEGORY_NUM[info_from_filename[2]], # Category Number
                    info_from_filename[2],               # Category
                    info_from_filename[0],               # Author ID
                    author_text,                         # Text
                ])
            except TypeError:
                with print_lock:
                    print(
                        'Type error when processing Author:\n'
                        f'\tid: {info_from_filename[0]}\n'
                        f'\tfile: {author_file_path}'
                    )
                    print('Skipping this author.')
        else:
            with print_lock:
                print(f'{author_file_path} is not an xml file!')
    csv_file.close()

    with print_lock:
        print(f'Finished making \'{csv_file_name}\'.')

#
def grabAuthors(csv_path, xml_path, print_lock):

    with print_lock:
        print('Getting Authors...')
    author_files = os.listdir(data_path)

    file_num = 0
    running = True
    while running:
        if file_num > len(author_files):
            running = False
        else:
            if file_num +BATCH_SIZE > len(author_files):
                xml_file_names = author_files[file_num :]
            else:
                xml_file_names = author_files[file_num : file_num +BATCH_SIZE]

            createCSVFile(
                xml_path,
                xml_file_names,
                csv_path,
                f'Authors_{file_num :06d}-{file_num +BATCH_SIZE -1 :06d}.csv',
                print_lock
            )
            file_num += BATCH_SIZE

#
def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass the following in order:\n'
            '\tThe path in which to place the CSV files.\n'
            '\tThe path in which the xml data is.'
        )
        sys.exit(0)

    output_path = sys.argv[1]
    data_path = sys.argv[2]

    return output_path, data_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    print_lock = Lock()

    output_path, data_path = grabArguments()
    grabAuthors(output_path, data_path, print_lock)
