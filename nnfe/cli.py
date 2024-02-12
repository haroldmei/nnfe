"""Console script for nnfe."""
import argparse
import sys

import pandas as pd
from nnfe.aggregation import make_nn_feature

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
    parser.add_argument('-e', '--eid', type=str, default = 'entity_id', help="specify the entity id column")
    parser.add_argument('-t', '--tid', type=str, default = 'time_id', help="specify the time id column")
    parser.add_argument('-l', '--leakage', action=argparse.BooleanOptionalAction, help="Allow data leakage")
    parser.add_argument('-r', '--reference', type=str, help="specify the reference columns to be used for finding nearest neighbors")
    parser.add_argument('-f', '--feature', type=str, help="specify the feature to be used for aggregating nn features")
    parser.add_argument('-a', '--aggregation', type=str, help="the aggregation function to be used")
    args = parser.parse_args()

    print("Arguments: ", args)
    
    df = pd.read_pickle(f'{args.input}')
    df = make_nn_feature(df, args.eid, args.tid, args.target, args.feature, args.aggregation, args.leakage)
    
    df.to_pickle(f'{args.output}')
    print(f'output file is saved at {args.output}')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
