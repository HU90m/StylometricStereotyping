#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
CHECK_DISTRIBUTIONS = True
UP_TO_CATEGORY = 4


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path
import numpy as np

if CHECK_DISTRIBUTIONS:
    from matplotlib import pyplot as plt


#---------------------------------------------------------------------------
# Globals
#---------------------------------------------------------------------------
#
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
CATEGORY_BETA = {
    0 : (10, 2),
    1 : (2, 10),
    2 : (4, 2),
    3 : (2, 4),
    4 : (1, 1),
    5 : (1, 1),
}


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
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

    categories_file = path.join(vector_dir, 'categories.npy')
    reliabilities_file = path.join(vector_dir, 'reliabilities')

    print('Loading Categories...')
    categories = np.load(categories_file, allow_pickle=True)

    print('Generating Reliabilities...')
    reliabilities = np.array(
        [np.random.beta(*CATEGORY_BETA[item]) for item in categories]
    )

    print('Saving Reliabilities...')
    np.save(reliabilities_file, reliabilities)


    if CHECK_DISTRIBUTIONS:
        print('Preparing Histogram...')

        category_selections = [
            categories == cat_num for cat_num in range(UP_TO_CATEGORY)
        ]
        seperated_reliabilities = [
            reliabilities[selection] for selection in category_selections
        ]

        print('Displaying Histogram...')
        for cat_num in range(UP_TO_CATEGORY):
            plt.hist(
                x=seperated_reliabilities[cat_num],
                bins='auto',
                alpha=0.6,
                label=CATEGORY_NAME[cat_num],
            )
        plt.xlabel('Distributions')
        plt.legend()
        plt.show()

    print('All Done.')
