from diffusion_uncertainty.paths import FID, RESULTS
import argparse
import yaml
import json
import pandas as pd

def main():
    fid_score_json = RESULTS.joinpath('fid_scores.json')

    with open(fid_score_json, 'r') as f:
        fid_score_dict = json.load(f)

    df = pd.DataFrame(fid_score_dict)

    df = df.query('with_uncertainty < 20')

    pivot_table = df.pivot_table(index=['dataset'], columns=['scheduler_type'], values='delta_fid', aggfunc='mean')

    print(pivot_table)

    pivot_table.to_csv(RESULTS.joinpath('delta_fid_scores.csv'))

if __name__ == '__main__':
    main()