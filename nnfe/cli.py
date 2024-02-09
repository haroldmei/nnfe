"""Console script for nnfe."""
import argparse
import sys

import pandas as pd
from nnfe.neighbor import make_nn_feature

def main():
    """Console script for nnfe.

    This function is the entry point of the nnfe command-line interface.
    It parses the command-line arguments, reads the input file, applies
    the make_nn_feature function to the data, and saves the output file.

    Returns:
        int: The exit code of the function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="specify the input file path")
    parser.add_argument('-o', '--output', type=str, help="specify the output file path")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    
    df = pd.read_pickle(f'{args['input']}')
    df = make_nn_feature(df)
    
    df.to_pickle(f'{args['output']}')
    print(f'output file is saved at {args['output']}')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
