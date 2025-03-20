import gc
from pathlib import Path
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

def plot_uncertainty(uncertainty: torch.Tensor, title: str, output_path: Path):
    """Plot and save uncertainty data."""
    mean_uncertainty = uncertainty.sum(dim=(-1, -2, -3)).mean(dim=0).flatten()
    std_uncertainty = uncertainty.sum(dim=(-1, -2, -3)).std(dim=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_uncertainty)
    plt.fill_between(range(len(mean_uncertainty)), 
                     mean_uncertainty - std_uncertainty, 
                     mean_uncertainty + std_uncertainty, 
                     alpha=0.3)
    fontsize_text = 20
    plt.xlabel('Sampling step ($t$)', fontsize=fontsize_text)
    plt.ylabel('Uncertainty', fontsize=fontsize_text)
    plt.title(title, fontsize=fontsize_text)
    plt.yticks([])
    plt.xticks([0, 10, 20, 30, 40, 50][::-1], fontsize=fontsize_text-4)
    plt.savefig(output_path)
    plt.close()

    print(f"Mean uncertainty: {mean_uncertainty}")
    print(f"Std uncertainty: {std_uncertainty}")

@torch.no_grad()
def main():
    df_runs = get_df_runs()



    # Process ImageNet512 (latent space)
    uncertainty_imagenet512 = load_uncertainty_data(df_runs, "imagenet512")
    if uncertainty_imagenet512 is not None:
        plot_uncertainty(uncertainty_imagenet512, 
                         'Latent space uncertainty', 
                         PLOT / 'uncertainty_schedule_latent_space.png')
        print('Printed latent space uncertainty')
        del uncertainty_imagenet512
    else:
        print('No valid data found for ImageNet512')

    # Process ImageNet64 (pixel space)
    uncertainty_imagenet64 = load_uncertainty_data(df_runs, "imagenet64", "uncertainty_centered")
    if uncertainty_imagenet64 is not None:
        plot_uncertainty(uncertainty_imagenet64, 
                         'Pixel space uncertainty', 
                         PLOT / 'uncertainty_schedule_pixel_space.png')
        print('Printed pixel space uncertainty')
        del uncertainty_imagenet64
    else:
        print('No valid data found for ImageNet64')

    # Process ImageNet256 (latent space)
    uncertainty_imagenet256 = load_uncertainty_data(df_runs, "imagenet256")
    if uncertainty_imagenet256 is not None:
        plot_uncertainty(uncertainty_imagenet256, 
                         'Latent space uncertainty', 
                         PLOT / 'uncertainty_schedule_latent_space_imagenet256.png')
        print('Printed latent space uncertainty for ImageNet256')
        del uncertainty_imagenet256
    else:
        print('No valid data found for ImageNet256')


    # Process imagenet128 (pixel space)
    uncertainty_imagenet128 = load_uncertainty_data(df_runs, "imagenet128")
    if uncertainty_imagenet128 is not None:
        plot_uncertainty(uncertainty_imagenet128, 
                         'Pixel space uncertainty', 
                         PLOT / 'uncertainty_schedule_pixel_space_imagenet128.png')
        print('Printed pixel space uncertainty for ImageNet128')
        del uncertainty_imagenet128
    else:
        print('No valid data found for ImageNet128')

    print('Done!')
    gc.collect()

    # Process cifar10 (pixel space)
    uncertainty_cifar10 = load_uncertainty_data(df_runs, "cifar10")
    if uncertainty_cifar10 is not None:
        plot_uncertainty(uncertainty_cifar10, 
                         'Pixel space uncertainty', 
                         PLOT / 'uncertainty_schedule_pixel_space_cifar10.png')
        print('Printed pixel space uncertainty')
    else:
        print('No valid data found for CIFAR10')

if __name__ == '__main__':
    main()