import sys
import os
from os.path import isfile, join, splitext

from multiprocessing import Process, Manager

import csv

from xml.etree import ElementTree
from html.parser import HTMLParser

import re


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
MULTIPROCESSING = True
KEEP_EXPRESSION = '[\w ]+'


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
    conversations = root[0]

    for conversation in conversations:
        if isinstance(conversation.text, str):
            html2text.feed(conversation.text)
        else:
            return False

    return ''.join(keep_expr.findall(html2text.text.lower()))

#
def createCSVFile(
    xml_path,
    xml_file_names,
    csv_path,
    csv_file_prefix,
    print_lock
):
    category_counts = [0, 0, 0, 0, 0, 0]
    keep_expression = re.compile(KEEP_EXPRESSION)

    csv_file_name = csv_file_prefix + '.csv'
    with print_lock:
        print(f'Making \'{csv_file_name}\'.')
    csv_file_location = join(csv_path, csv_file_name)
    csv_file = open(csv_file_location, 'w')
    csv_writer = csv.writer(
        csv_file,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )
    for file_num, author_file in enumerate(xml_file_names):

        author_file_location = join(xml_path, author_file)
        file_no_ext, file_ext = splitext(author_file)

        info_from_filename = file_no_ext.split('_', 2)

        if isfile(author_file_location) and (file_ext == '.xml'):
            author_text = grabAuthorText(
                author_file_location,
                keep_expression,
            )
            if author_text:
                category_num = CATEGORY_NUM[info_from_filename[2]]
                csv_writer.writerow([
                    category_num,          # Category Number
                    info_from_filename[2], # Category
                    info_from_filename[0], # Author ID
                    author_text,           # Text
                ])
                category_counts[category_num] += 1
            else:
                with print_lock:
                    print(
                        'Author had NoneType in conversation:\n'
                        f'\tid: {info_from_filename[0]}\n'
                        f'\tfile: {author_file_location}'
                    )
                    print('Skipping this author.')
        else:
            with print_lock:
                print(f'{author_file_location} is not an xml file!')
    csv_file.close()

    with print_lock:
        print(f'Finished making \'{csv_file_name}\'.')

    csv_file_name_with_info = csv_file_prefix
    for category_count in category_counts:
        csv_file_name_with_info += '_' + str(category_count)
    csv_file_name_with_info += '.csv'
    csv_file_location_with_info = join(csv_path, csv_file_name_with_info)
    os.rename(
        csv_file_location,
        csv_file_location_with_info,
    )

    with print_lock:
        print(
            f'Moved \'{csv_file_name}\' to',
            f'\'{csv_file_location_with_info}\'.',
        )



#
def grabAuthors(csv_path, xml_path, print_lock):

    with print_lock:
        print('Getting Authors...')

    xml_file_names = os.listdir(xml_path)
    len_xml_file_names = len(xml_file_names)

    arg_list = []

    num_csv_files = (len_xml_file_names // BATCH_SIZE) + 1

    for idx in range(0, num_csv_files -1):
        start_idx = idx*BATCH_SIZE
        end_idx   = idx*BATCH_SIZE +BATCH_SIZE -1
        arg_list.append((
            xml_path,
            xml_file_names[start_idx:end_idx],
            csv_path,
            f'Authors_{start_idx :06d}-{end_idx :06d}',
            print_lock
        ))

    start_idx = (num_csv_files -1)*BATCH_SIZE
    arg_list.append((
        xml_path,
        xml_file_names[start_idx : len_xml_file_names],
            csv_path,
            f'Authors_{start_idx :06d}-{len_xml_file_names :06d}',
            print_lock
    ))

    if MULTIPROCESSING:
        processes = []
        for idx, arguments in enumerate(arg_list):
            processes.append(
                Process(target=createCSVFile, args=arguments)
            )
            processes[idx].start()

        for process in processes:
            process.join()
    else:
        for arguments in arg_list:
            createCSVFile(*arguments)

#
def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass the following in order:\n'
            '\tThe path in which the xml data is.'
            '\tThe path in which to place the CSV files.\n'
        )
        sys.exit(0)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    return output_path, data_path


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    manager = Manager()
    print_lock = manager.Lock()

    output_path, data_path = grabArguments()
    grabAuthors(output_path, data_path, print_lock)
