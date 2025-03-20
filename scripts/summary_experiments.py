#!/usr/bin/env python3

import argparse
import sys
from path import Path
import yaml

sys.path.append(Path(__file__).absolute().parent.parent)

import pandas as pd
from diffusion_uncertainty.paths import FID, RESULTS

def get_extra(row):
    if row['scheduler_type'] == 'uncertainty_zigzag_centered':
        return f'zigzag={int(row["num_zigzag"])}'
    return ''

def parse_args():
    argparser = argparse.ArgumentParser()

    filter_group = argparser.add_argument_group('Filters')
    filter_group.add_argument('--scheduler-type', type=str, help='Filter by scheduler type', dest='filter_scheduler_type')  
    filter_group.add_argument('--dataset', type=str, help='Filter by dataset', dest='filter_dataset')
    filter_group.add_argument('--start-index', type=int, help='Filter by start index', dest='filter_start_index')
    filter_group.add_argument('--generation-steps', type=int, help='Filter by generation steps', dest='filter_generation_steps')
    filter_group.add_argument('--num-samples', type=int, help='Filter by num samples', dest='filter_num_samples')
    filter_group.add_argument('--M-greater-eq-than', type=int, help='Filter by M greater or equal than', dest='filter_M_greater_eq_than')

    return argparser.parse_args()


def filter_df(df, args):
    """
    Filter a DataFrame based on the provided arguments.

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        args (argparse.Namespace): The arguments containing the filter conditions.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    if args.filter_scheduler_type:
        df = df[df['scheduler_type'] == args.filter_scheduler_type]
        print('Filtered by scheduler type:', args.filter_scheduler_type)
    if args.filter_dataset:
        df = df[df['dataset'] == args.filter_dataset]
        print('Filtered by dataset:', args.filter_dataset)
    if args.filter_start_index:
        df = df[df['start_index'] == args.filter_start_index]
        print('Filtered by start index:', args.filter_start_index)
    if args.filter_generation_steps:
        df = df[df['generation_steps'] == args.filter_generation_steps]
        print('Filtered by generation steps:', args.filter_generation_steps)
    if args.filter_num_samples:
        df = df[df['num_samples'] == args.filter_num_samples]
        print('Filtered by num samples:', args.filter_num_samples)
    if args.filter_M_greater_eq_than:
        df = df[df['M'] >= args.filter_M_greater_than]
        print('Filtered by M greater than:', args.filter_M_greater_than)
    return df

def main():
    args = parse_args()
    df = get_df_runs()
    df = filter_df(df, args)
    df['extra'] = df.apply(get_extra, axis=1)
    print('=============')
    # print(df)
    
    # Set the print options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    # Disable maximum row length
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    COLUMNS_TO_DISPLAY = ['scheduler_type', 'extra', 'dataset', 'start_index', 'folder-name', 'generation_steps', 'num_samples', 'M']
    print(df[COLUMNS_TO_DISPLAY].sort_values(by=['scheduler_type', 'dataset', 'start_index', 'folder-name',]))

    print('=============')

    df_grouped = df.groupby(['scheduler_type', 'dataset'])[['num_samples']].sum()
    print(df_grouped)

    print('=============================')

    df_grouped_2 = df.groupby(['scheduler_type', 'extra', 'dataset'])['folder-name'].agg(lambda x: ' '.join(sorted(x)))
    print(df_grouped_2)

def get_df_runs():
    score_uncertainty = RESULTS / 'score-uncertainty'
    configs = []
    for folder in score_uncertainty.dirs():
        config_path = folder / 'args.yaml'
        if not config_path.exists():
            print('Skipping', folder.name, 'because args.yaml does not exist')
            continue
        pth_files = folder.files('*.pth')
        if len(pth_files) == 0:
            print('Skipping', folder.name, 'because no .pth files exist')
            continue
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['folder-name'] = folder.name
        if 'scheduler_type' not in config:
            config['scheduler_type'] = 'mc_dropout'
        if config['scheduler_type'] == 'mc_dropout':
            if config['dropout'] != 0.5: 
                print('Skipping', folder.name, 'because dropout is not 0.5')
                continue 
        configs.append(config)
    df = pd.DataFrame(configs)
    return df

if __name__ == '__main__':
    main()
