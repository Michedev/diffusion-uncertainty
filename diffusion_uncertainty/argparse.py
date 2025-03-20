import argparse


def add_scheduler_uncertainty_args_(argparser: argparse.ArgumentParser):
    """
    Add uncertainty-related arguments to the given argument parser.

    Args:
        argparser (argparse.ArgumentParser): The argument parser to add arguments to.

    The function adds the following arguments:
    - M (int): Number of Monte Carlo samples (default: 30)
    - start_step_uc (int): Starting step for uncertainty calculation (default: 0)
    - num_steps_uc (int): Number of steps for uncertainty calculation (default: 20)
    - seed (int): Random seed for reproducibility (default: 38482234)
    - eta (float): Eta parameter for uncertainty calculation (default: 0.00)
    - dropout (float): Dropout rate for MC dropout (default: 0.1)
    - scheduler_type (str): Type of scheduler to use (default: 'mc_dropout')
    - start_index (int): Starting index for processing (default: 0)

    Additional argument groups:
    1. uncertainty:
       - predict_next (bool): Whether to predict the next step

    2. uncertainty_distance:
       - uncertainty_distance (int): Distance for uncertainty calculation (default: 20)

    3. uncertainty_zigzag_centered:
       - num_zigzag (int): Number of zigzags for the centered noise inference (default: 3)
    """
    
    argparser.add_argument('-M', type=int, default=30, dest='M')
    argparser.add_argument('--start-step-uc', '--start-step', type=int, default=0, dest='start_step_uc')
    argparser.add_argument('--num-steps-uc', type=int, default=20, dest='num_steps_uc')
    argparser.add_argument('--seed', type=int, default=38482234)
    argparser.add_argument('--eta', type=float, default=0.00)
    argparser.add_argument('--dropout', type=float, default=0.1, dest='dropout')
    argparser.add_argument('--scheduler-type', '--scheduler', type=str, default='mc_dropout', choices=['mc_dropout', 'flip', 'uncertainty', 'flip_grad', 'uncertainty_single', 'uncertainty_single_score', 'uncertainty_centered', 'uncertainty_centered_d', 'uncertainty_image', 'uncertainty_original', 'uncertainty_zigzag_centered', 'uncertainty_fisher'], dest='scheduler_type')
    argparser.add_argument('--start-index', type=int, default=0, required=False)

    uncertainty_args = argparser.add_argument_group('uncertainty')
    uncertainty_args.add_argument('--predict-next', action='store_true', dest='predict_next')

    infere_noise_distance_args = argparser.add_argument_group('uncertainty_distance')
    infere_noise_distance_args.add_argument('--uncertainty-distance', type=int, default=20, dest='uncertainty_distance')

    uncertainty_zigzag_centered_args = argparser.add_argument_group('uncertainty_zigzag_centered')
    uncertainty_zigzag_centered_args.add_argument('--num-zigzag', '--num-zigzags', '--num-zig-zag', '--num-zig-zags', type=int, default=3, dest='num_zigzag')