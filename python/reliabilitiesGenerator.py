#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
CHECK_DISTRIBUTIONS = True

IS_PAN13 = False



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
if IS_PAN13:
    CATEGORY_NAME = {
        0 : '30s_female',
        1 : '30s_male',
        2 : '20s_female',
        3 : '20s_male',
    }
    CATEGORY_BETAS = {
        'little' : {
            0 : (80, 20),
            1 : (20, 80),
            2 : (60, 40),
            3 : (40, 60),
        },
        'some' : {
            0 : (40, 10),
            1 : (10, 40),
            2 : (30, 20),
            3 : (20, 30),
        },
        'much' : {
            0 : (20,  5),
            1 : ( 5, 20),
            2 : (15, 10),
            3 : (10, 15),
        },
    }
else:
    CATEGORY_NAME = {
        0 : 'bot',
        1 : 'human',
    }
    CATEGORY_BETAS = {
        'little' : {
            0 : (40, 60),
            1 : (60, 40),
        },
        'some' : {
            0 : (20, 30),
            1 : (30, 20),
        },
        'much' : {
            0 : (20, 30),
            1 : (30, 20),
        },
    }


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def grabArguments():
    if len(sys.argv) < 2:
        print('Please pass the vectors directory.')
        sys.exit(0)

    if not path.isdir(sys.argv[1]):
        print(f'The following is not a directory:\n\t{sys.argv[1]}')
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

    print('Loading Categories...')
    categories = np.load(categories_file, allow_pickle=True)

    reliabilities = {}

    for distribution_name, category_beta in CATEGORY_BETAS.items():
        reliabilities_file = path.join(
            vector_dir,
            'reliabilities_' + distribution_name,
        )

        print(f'Generating \'{distribution_name}\' Reliabilities...')
        reliabilities[distribution_name] = np.array(
            [np.random.beta(*category_beta[item]) for item in categories]
        )

        print(f'Saving \'{distribution_name}\' Reliabilities...')
        np.save(reliabilities_file, reliabilities[distribution_name])


    if CHECK_DISTRIBUTIONS:

        print('Preparing Histogram...')

        fig, plots = plt.subplots(nrows=len(CATEGORY_BETAS), ncols=1)

        category_selections = [
            categories == cat_num for cat_num in range(len(CATEGORY_NAME))
        ]

        for idx, distribution_name in enumerate(CATEGORY_BETAS):

            seperated_reliabilities = [
                reliabilities[distribution_name][selection]
                for selection in category_selections
            ]

            for cat_num in range(len(CATEGORY_NAME)):
                plots[idx].hist(
                    x=seperated_reliabilities[cat_num],
                    bins='auto',
                    alpha=0.6,
                    label=CATEGORY_NAME[cat_num],
                )
            plots[idx].set_title(distribution_name)
            plots[idx].legend()

        print('Displaying Histogram...')
        plt.show()

    print('All Done.')
