#!/bin/python3
#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
IS_PAN13 = False
IS_GENDER = True
BATCH_SIZE = 10000
MULTIPROCESSING = False

MAKE_LOWERCASE = True
SELECT_CHARACTERS = False
KEEP_EXPRESSION = '[\w ,!?&:\-\'\"]+'


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
import os
from os import path

from multiprocessing import Process, Manager

import re

import csv
from xml.etree import ElementTree
from html.parser import HTMLParser


#---------------------------------------------------------------------------
# Globals
#---------------------------------------------------------------------------
#
if IS_PAN13:
    CATEGORY_NUM = {
        '30s_female' : 0,
        '30s_male'   : 1,
        '20s_male'   : 2,
        '20s_female' : 3,
        '10s_female' : 4,
        '10s_male'   : 5,
    }
    CATEGORY_NAME = {
        0 : '30s_female',
        1 : '30s_male',
        2 : '20s_female',
        3 : '20s_male',
        4 : '10s_female',
        5 : '10s_male',
    }
    NUM_CATEGORIES = 6
    IGNORED_CATEGORIES = (4, 5)
elif IS_GENDER:
    CATEGORY_NUM = {
        'male'   : 0,
        'female' : 1,
    }
    CATEGORY_NAME = {
        0 : 'male',
        1 : 'female',
    }
    NUM_CATEGORIES = 2
    IGNORED_CATEGORIES = ()
else:
    CATEGORY_NUM = {
        'bot'   : 0,
        'human' : 1,
    }
    CATEGORY_NAME = {
        0 : 'bot',
        1 : 'human',
    }
    NUM_CATEGORIES = 2
    IGNORED_CATEGORIES = ()


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
def grabAuthorText(
    author_file_path,
    keep_expr,
):
    html2text = HTML2Text()

    tree = ElementTree.parse(
        author_file_path,
        parser=ElementTree.XMLParser(encoding="utf-8"),
    )
    root = tree.getroot()
    bodies = root[0]

    for body in bodies:
        if isinstance(body.text, str):
            html2text.feed(body.text)
            html2text.feed(' ')
        else:
            return False

    author_text = html2text.text

    if MAKE_LOWERCASE:
        author_text = author_text.lower()

    if SELECT_CHARACTERS:
        author_text = ''.join(keep_expr.findall(author_text))

    return author_text

#
def createCSVFile(
    xml_path,
    csv_path,
    author_ids,
    author_categories,
    author_filenames,
    csv_file_prefix,
    print_lock
):
    category_counts = [0]*NUM_CATEGORIES
    keep_expression = re.compile(KEEP_EXPRESSION)

    csv_filename = csv_file_prefix + '.csv'
    with print_lock:
        print(f'Making \'{csv_filename}\'.')

    csv_file_location = path.join(csv_path, csv_filename)
    csv_file = open(csv_file_location, 'w')
    csv_writer = csv.writer(
        csv_file,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )
    for author_idx, author_filename in enumerate(author_filenames):

        author_file_location = path.join(xml_path, author_filename)
        category_num = CATEGORY_NUM[author_categories[author_idx]]

        if path.isfile(author_file_location):
            if category_num not in IGNORED_CATEGORIES:
                author_text = grabAuthorText(
                    author_file_location,
                    keep_expression,
                )
                if author_text:
                    csv_writer.writerow([
                        category_num,                  # Category Number
                        author_categories[author_idx], # Category
                        author_ids[author_idx],        # Author ID
                        author_text,                   # Text
                    ])
                    category_counts[category_num] += 1
                else:
                    with print_lock:
                        print(
                            'Author had NoneType in text body:\n'
                            f'\tfile: {author_file_location}'
                        )
                        print('Skipping this author.')
        else:
            with print_lock:
                print(f'{author_file_location} is not an xml file!')
    csv_file.close()

    with print_lock:
        print(f'Finished making \'{csv_filename}\'.')

    csv_file_name_with_info = csv_file_prefix + '_' + str(sum(category_counts))
    for category_count in category_counts:
        csv_file_name_with_info += '-' + str(category_count)
    csv_file_name_with_info += '.csv'
    csv_file_location_with_info = path.join(csv_path, csv_file_name_with_info)
    os.rename(
        csv_file_location,
        csv_file_location_with_info,
    )
    with print_lock:
        print(
            f'Moved \'{csv_filename}\' to',
            f'\'{csv_file_location_with_info}\'.',
        )

#
def grabAuthors(truth_file, xml_path, csv_path, print_lock):
    with print_lock:
        print('Getting Authors...')

    author_ids = []
    author_categories = []
    author_filenames = []

    if IS_PAN13:
        for author_truth in open(truth_file, 'r'):
            author_truth = author_truth[:-1]
            author_info = author_truth.split(':::')
            if len(author_info) > 2:
                author_filename = '_'.join((
                    author_info[0],
                    'en',
                    author_info[2],
                    author_info[1],
                )) + '.xml'

                author_ids.append(author_info[0])
                author_categories.append(author_info[2] + '_' + author_info[1])
                author_filenames.append(author_filename)
    elif IS_GENDER:
        for author_truth in open(truth_file, 'r'):
            author_truth = author_truth[:-1]
            author_info = author_truth.split(':::')
            if len(author_info) > 2 and author_info[1] == 'human':
                author_ids.append(author_info[0])
                author_categories.append(author_info[2])
                author_filenames.append(author_info[0] + '.xml')
    else:
        for author_truth in open(truth_file, 'r'):
            author_truth = author_truth[:-1]
            author_info = author_truth.split(':::')
            if len(author_info) > 2:
                author_ids.append(author_info[0])
                author_categories.append(author_info[1])
                author_filenames.append(author_info[0] + '.xml')


    num_authors = len(author_filenames)
    num_csv_files = (num_authors // BATCH_SIZE) + 1

    argument_lists = []
    for idx in range(0, num_csv_files -1):
        start_idx = idx*BATCH_SIZE
        end_idx   = idx*BATCH_SIZE +BATCH_SIZE -1
        argument_lists.append((
            xml_path,
            csv_path,
            author_ids[start_idx:end_idx],
            author_categories[start_idx:end_idx],
            author_filenames[start_idx:end_idx],
            f'Authors_{start_idx :06d}-{end_idx :06d}',
            print_lock,
        ))
    start_idx = (num_csv_files -1)*BATCH_SIZE
    argument_lists.append((
        xml_path,
        csv_path,
        author_ids[start_idx:num_authors],
        author_categories[start_idx:num_authors],
        author_filenames[start_idx:num_authors],
        f'Authors_{start_idx :06d}-{num_authors :06d}',
        print_lock,
    ))

    if MULTIPROCESSING:
        processes = []
        for idx, arguments in enumerate(argument_lists):
            processes.append(
                Process(target=createCSVFile, args=arguments)
            )
            processes[idx].start()

        for process in processes:
            process.join()
    else:
        for arguments in argument_lists:
            createCSVFile(*arguments)

#
def grabArguments():
    if len(sys.argv) < 4:
        print(
            'Please pass the following in order:\n'
            '\tThe PAN truth file.\n'
            '\tThe path in which the xml data is.\n'
            '\tThe path in which to place the CSV files.'
        )
        sys.exit(0)

    if not path.isfile(sys.argv[1]):
        print(f'The following is not a file:\n\t{sys.argv[1]}')
        sys.exit(0)

    if not path.isdir(sys.argv[2]):
        print(f'The following is not a directory:\n\t{sys.argv[2]}')
        sys.exit(0)

    if not path.isdir(sys.argv[3]):
        print(f'The following is not a directory:\n\t{sys.argv[3]}')
        sys.exit(0)

    truth_file = sys.argv[1]
    xml_path = sys.argv[2]
    csv_path = sys.argv[3]

    return truth_file, xml_path, csv_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    manager = Manager()
    print_lock = manager.Lock()

    truth_file, csv_path, xml_path = grabArguments()

    grabAuthors(truth_file, csv_path, xml_path, print_lock)
