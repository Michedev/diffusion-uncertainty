import argparse
import gc
from pathlib import Path
from random import randint
import random
import torch
import matplotlib.pyplot as plt
from diffusion_uncertainty.paths import SCORE_UNCERTAINTY, PLOT
from scripts.summary_experiments import get_df_runs

def load_uncertainty_data(df, dataset_name, scheduler_type=None, limit=3):
    """Load uncertainty data for a given dataset."""
    uncertainty_data = []
    query = f'dataset == "{dataset_name}" and generation_steps == 50 and num_samples >= 1000'
    if scheduler_type:
        query += f' and scheduler_type == "{scheduler_type}"'
    
    df_filtered = df.query(query)
    
    for folder in df_filtered['folder-name']:
        run_path = SCORE_UNCERTAINTY / folder
        for uncertainty_path in run_path.glob('uncertainty*.pth'):
            uncertainty = torch.load(uncertainty_path).cpu()
            if uncertainty.shape[1] == 50:
                uncertainty_data.append(uncertainty)
                break  # Only take the first valid uncertainty file per run
        
        if len(uncertainty_data) == limit:
            break
    
    uncertainty_data = torch.cat(uncertainty_data, dim=0) if uncertainty_data else None
    if uncertainty_data is not None:
        print(f'Loaded uncertainty data for {dataset_name} with shape {uncertainty_data.shape}')
    return uncertainty_data

def plot_uncertainty(uncertainty: torch.Tensor, image: torch.Tensor, title: str, output_path: Path):
    """Plot and save uncertainty data."""
    uncertainty = uncertainty[::5]
    fig, axs = plt.subplots(1, len(uncertainty)+1, figsize=(5 * len(uncertainty), 6))

    for i in range(len(uncertainty)):
        uncertainty_i = uncertainty[i]
        uncertainty_i = uncertainty_i.amax(dim=0, keepdim=True)
        uncertainty_i = uncertainty_i.permute(1, 2, 0)
        uncertainty_i = (uncertainty_i - uncertainty_i.min()) / (uncertainty_i.max() - uncertainty_i.min())
        if title == 'Our method':
            mask = uncertainty_i > 0.5
            uncertainty_i[mask] = uncertainty_i[mask] ** 1.25
            uncertainty_i[~mask] = uncertainty_i[~mask] ** 0.75

            # mask = uncertainty_i < 0.05
            # uncertainty_i[mask] = 1 - uncertainty_i[mask] 
        axs[i].imshow(uncertainty_i, cmap='coolwarm')

        axs[i].set_xlabel(f'Sampling step ($t={(len(uncertainty)-i-1)*5}$)', fontsize=20)
        axs[i].set_yticks([])
        axs[i].set_xticks([])
    axs[-1].imshow(image.permute(1, 2, 0))
    axs[-1].set_xlabel('Generated image', fontsize=20)
    axs[-1].set_yticks([])
    axs[-1].set_xticks([])

    fig.savefig(output_path, bbox_inches='tight', )
    fig.tight_layout() 
    plt.close(fig)

@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--scheduler', type=str, default='mc_dropout', help='The scheduler type to plot.')

    args = argparser.parse_args()

    df_runs = get_df_runs()
    df_filtered = df_runs.query(f'scheduler_type == "{args.scheduler}"')

    mc_dropout_plot = PLOT / args.scheduler
    mc_dropout_plot.mkdir_p()

    random_uncertainty = []
    random_image = []
    k = 0
    for folder in df_filtered['folder-name']:
        run_path = SCORE_UNCERTAINTY / folder
        for uncertainty_path in run_path.glob('uncertainty*.pth'):
            uncertainty = torch.load(uncertainty_path).cpu()
            if uncertainty.shape[1] < 20:
                continue
            image_pack = torch.load(uncertainty_path.parent / uncertainty_path.basename().replace(f'uncertainty_{args.scheduler}', 'gen_images')).cpu()
            print('loaded from', uncertainty_path)
            for _ in range(3):
                i_random = randint(0, uncertainty.shape[0] - 1)
                random_uncertainty.append(torch.clone(uncertainty[i_random]))
                random_image.append(torch.clone(image_pack[i_random]))
                if k == 5: 
                    break
                k += 1
            if len(random_uncertainty) >= 5:
                break
            del uncertainty, image_pack
            gc.collect()
            if k == 5: break
        if k == 5: break
        if len(random_uncertainty) >= 5:
            break
    print('start plotting')
    title = 'MC-Dropout' if args.scheduler == 'mc_dropout' else 'Our method'
    for i, sequence in enumerate(random_uncertainty):
        plot_uncertainty(sequence, random_image[i], title, mc_dropout_plot / f'random_uncertainty_{i}.png')



if __name__ == '__main__':
    main()