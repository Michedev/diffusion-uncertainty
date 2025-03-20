from diffusion_uncertainty.paths import RESULTS
from path import Path


def clean_empty_runs():
    SCORE_UNCERTAINTY = RESULTS / 'score-uncertainty'
    for run in SCORE_UNCERTAINTY.dirs():
        if not any(list(run.files('*.pth'))):
            print(f"Removing empty run: {run}")
            run.rmtree()

if __name__ == '__main__':
    clean_empty_runs()