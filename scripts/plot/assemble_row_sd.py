import argparse
from diffusion_uncertainty.paths import STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE
from torchvision.io import read_image
from torchvision.utils import save_image, make_grid

def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('folders', type=str, nargs='+', help='Stable Diffusion folders')

    args = argparser.parse_args()

    images = []
    first = True
    original_image = None
    for folder in args.folders:
        print('Processing', folder)
        folder = STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / folder

        assert ((folder / 'args.yaml').exists())
        assert ((folder / 'output_sd_uc.png').exists())

        if first:
            original_image = read_image(folder / 'output_sd.png')
            original_image = original_image.float() / 255.0
            first = False
            images.append(original_image)
        else:
            org_image = read_image(folder / 'output_sd.png')
            org_image = org_image.float() / 255.0
            # assert ((org_image - original_image).abs() < 1e-1).all()

        uc_img = read_image(folder / 'output_sd_uc.png')

        uc_img = uc_img.float() / 255.0

        images.append(uc_img)

    grid_images = make_grid(images, nrow=len(images), pad_value=1.0, padding=20)
    grid_folder = STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / 'grids'
    grid_folder.mkdir_p()
    i = 0
    while (grid_folder / f'{i}.png').exists():
        i += 1
    
    save_image(grid_images, grid_folder / f'{i}.png')

    print(f'Saved grid to {grid_folder / f"{i}.png"}')

if __name__ == '__main__':
    main()