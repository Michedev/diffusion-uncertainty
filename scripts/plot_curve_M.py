import matplotlib.pyplot as plt

from diffusion_uncertainty.paths import PLOT


def main():

    cifar10: list[float] = [13.406, 13.395, 13.395]
    Ms = [5, 10, 20]

    plt.plot(cifar10, 'o-', label='CIFAR-10')
    # Set y-axis ticks to show exact numbers
    plt.yticks(cifar10)

    plt.xlabel('M')
    plt.ylabel('FID Score')
    plt.title('Uncertainty low-quality filtering over M for CIFAR-10')
    plt.legend()
    plt.xticks(list(range(len(Ms))), Ms)

    dest_path = PLOT / 'times_over_M_cifar10.png'
    plt.savefig(dest_path)

    plt.close()

    print('Saved plot in', str(dest_path))

    cifar10: list[float] = [3.254, 3.248,  3.245]
    Ms = [5, 10, 20]

    plt.plot(cifar10, 'o-', label='Imagenet64')
    # Set y-axis ticks to show exact numbers
    plt.yticks(cifar10)

    plt.xlabel('M')
    plt.ylabel('FID Score')
    plt.title('Uncertainty low-quality filtering over M for ImageNet 64')
    plt.legend()
    plt.xticks(list(range(len(Ms))), Ms)

    dest_path = PLOT / 'times_over_M_imagenet64.png'
    plt.savefig(dest_path)

    plt.close()

    print('Saved plot in', str(dest_path))

    cifar10: list[float] = [7.8, 7.764,  7.754]
    Ms = [5, 10, 20]


    plt.plot(cifar10, 'o-', label='Imagenet256')
    # Set y-axis ticks to show exact numbers
    plt.yticks(cifar10)

    plt.xlabel('M')
    plt.ylabel('FID Score')
    plt.title('Uncertainty low-quality filtering over M for ImageNet 256')
    plt.legend()
    plt.xticks(list(range(len(Ms))), Ms)

    dest_path = PLOT / 'times_over_M_imagenet256.png'
    plt.savefig(dest_path)

    plt.close()

    print('Saved plot in', str(dest_path))



if __name__ == '__main__':
    main()


"""
python scripts/measure_times_imagenet.py --num-samples 128 --batch-size 128 -M 20 --start-step-uc 45 --num-steps-uc 3 --generation-steps 50 --scheduler uncertainty_centered --image-size 64 --start-index 0 --model-type unet ; python scripts/measure_times_imagenet.py --num-samples 128 --batch-size 128 -M 20 --start-step-uc 45 --num-steps-uc 3 --generation-steps 50 --scheduler uncertainty_centered --image-size 128 --start-index 0 --model-type unet; python scripts/measure_times_cifar10.py --num-samples 128 --batch-size 128 --generation-steps 50 -M 20 --start-step-uc 45 --num-steps-uc 3; python scripts/measure_times_imagenet.py --num-samples 128 --batch-size 128 -M 20 --start-step-uc 45 --num-steps-uc 3 --generation-steps 50 --scheduler dpm_2_uncertainty_centered --image-size 128 --start-index 0 --model-type unet; python scripts/measure_times_imagenet.py --num-samples 128 --batch-size 128 -M 20 --start-step-uc 45 --num-steps-uc 3 --generation-steps 50 --scheduler dpm_2_uncertainty_centered --image-size 256 --start-index 0 --model-type uvit;python scripts/measure_times_imagenet.py --num-samples 128 --batch-size 32 -M 20 --start-step-uc 45 --num-steps-uc 3 --generation-steps 50 --scheduler dpm_2_uncertainty_centered --image-size 512 --start-index 0 --model-type uvit;
"""