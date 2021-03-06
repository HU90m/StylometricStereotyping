#!/bin/python3
#---------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------
#
OUTPUT = False
FIND_DIVERGENCE = True
CHECK_DISTRIBUTIONS = True

IS_PAN13 = False
SEED=42
#SEED=665


#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------
#
import sys
from os import path
import numpy as np

if FIND_DIVERGENCE:
    from scipy.special import gamma, digamma

if CHECK_DISTRIBUTIONS:
    from scipy.stats import beta
    from matplotlib import pyplot as plt
    from matplotlib import rc
    rc('font', **{'family':'DejaVu Sans','serif':['Times'], 'size':8})
    rc('text', usetex=True)


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
    CATEGORY_COLOR = {
        0 : 'red',
        1 : 'blue',
        2 : 'orange',
        3 : 'cyan',
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
    PRINT_BETA = (
        'little',
        'some',
        'much',
    )
else:
    CATEGORY_NAME = {
        0 : 'Malicious',
        1 : 'Benign',
    }
    CATEGORY_COLOR = {
        0 : 'dimgrey',
        1 : 'darkturquoise',
    }
    CATEGORY_BETAS = {
        'vvvsimilar' : {
            0 : (5, 3),
            1 : (5, 2),
        },
        'vvsimilar' : {
            0 : (5, 4),
            1 : (5, 2),
        },
        'vsimilar' : {
            0 : (5, 5),
            1 : (5, 2),
        },
        'similar' : {
            0 : (3, 5),
            1 : (5, 2),
        },
        'different' : {
            0 : (2, 5),
            1 : (5, 2),
        },
        'vdifferent' : {
            0 : (4, 20),
            1 : (5, 2),
        },
        'vvdifferent' : {
            0 : (4, 30),
            1 : (5, 2),
        },
        'vvvdifferent' : {
            0 : (4, 40),
            1 : (5, 2),
        },
    }
    PRINT_BETA = {
        'vvvsimilar'   : '$p_1$',
        'vvsimilar'    : '$p_2$',
        'vsimilar'     : '$p_3$',
        'similar'      : '$p_4$',
        'different'    : '$p_5$',
        'vdifferent'   : '$p_6$',
        'vvdifferent'  : '$p_7$',
        'vvvdifferent' : '$p_8$',
    }


#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
#
def KLDiv(a_0, b_0, a_1, b_1):
    def B(a, b):
        return (gamma(a)*gamma(b))/gamma(a + b)
    return np.log(B(a_1, b_1)/B(a_0, b_0)) \
        - (a_1 - a_0)*digamma(a_0) \
        - (b_1 - b_0)*digamma(b_0) \
        + (a_1 - a_0 + b_1 - b_0)*digamma(a_0 + b_0)

def grabArguments():
    if len(sys.argv) < 3:
        print(
            'Please pass in order:\n'
            '\tThe input categories file.\n'
            '\tThe output reliabilities file prefix.\n'
        )
        sys.exit(0)

    if not path.isfile(sys.argv[1]):
        print(f'The following is not a directory:\n\t{sys.argv[1]}')
        sys.exit(0)

    categories_file = sys.argv[1]
    reliabilities_prefix = sys.argv[2]
    return categories_file, reliabilities_prefix


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
#
if __name__ == '__main__':
    categories_file, reliabilities_prefix = grabArguments()

    print('Loading Categories...')
    categories = np.load(categories_file, allow_pickle=True)

    np.random.seed(SEED)
    reliabilities = {}

    for distribution_name, category_beta in CATEGORY_BETAS.items():
        reliabilities_file = reliabilities_prefix + '_' + distribution_name

        print(f'Generating \'{distribution_name}\' Reliabilities...')
        reliabilities[distribution_name] = np.array(
            [np.random.beta(*category_beta[item]) for item in categories]
        )

        # https://math.wikia.org/wiki/Beta_distribution
        if FIND_DIVERGENCE:
            KL = KLDiv(
                category_beta[0][0],
                category_beta[0][1],
                category_beta[1][0],
                category_beta[1][1],
            )
            print(
                f"For '{distribution_name}', "
                f"the KL of '{CATEGORY_NAME[0]}' compared to "
                f"'{CATEGORY_NAME[1]}' is {KL}"
            )

        if OUTPUT:
            print(f'Saving \'{distribution_name}\' Reliabilities...')
            np.save(reliabilities_file, reliabilities[distribution_name])


    if CHECK_DISTRIBUTIONS:
        print('Preparing Histogram...')

        h_len = len(PRINT_BETA)//2
        print(len(PRINT_BETA))
        print(h_len)

        fig, plots = plt.subplots(ncols=h_len, nrows=2)

        category_selections = [
            categories == cat_num for cat_num in range(len(CATEGORY_NAME))
        ]
        for idx, distribution_name in enumerate(PRINT_BETA):
            seperated_reliabilities = [
                reliabilities[distribution_name][selection]
                for selection in category_selections
            ]

            for cat_num in range(len(CATEGORY_NAME)):
                col = idx % h_len
                row = int(idx > h_len -1)
                n, bins, patches = plots[row, col].hist(
                    x=seperated_reliabilities[cat_num],
                    bins=50,
                    alpha=0.6,
                    color=CATEGORY_COLOR[cat_num],
                    density=True,
                    stacked=True,
                )
                plots[row, col].plot(
                    bins,
                    beta.pdf(bins, *CATEGORY_BETAS[distribution_name][cat_num]),
                    color=CATEGORY_COLOR[cat_num],
                    label=CATEGORY_NAME[cat_num],
                )
            plots[row, col].set_title(PRINT_BETA[distribution_name])
            plots[row, col].set(xlim=(0,1))

        plots[1, h_len -1].legend()
        print('Displaying Histogram...')
        plt.tight_layout()
        plt.show()

    print('All Done.')
