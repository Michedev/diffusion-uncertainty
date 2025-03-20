import os
from re import I
from path import Path
from regex import B

ROOT = Path(__file__).parent.parent.parent

CONFIG = ROOT / 'config'
if not CONFIG.exists():
    CONFIG.mkdir()

RESULTS = ROOT / 'results'
if not RESULTS.exists():
    RESULTS.mkdir()

PR_MANIFOLD = RESULTS / 'pr_manifold'
if not PR_MANIFOLD.exists():
    PR_MANIFOLD.mkdir()

PRECISION_RECALL_CURVES = RESULTS / 'precision_recall_curves'
if not PRECISION_RECALL_CURVES.exists():
    PRECISION_RECALL_CURVES.mkdir()

STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE = RESULTS / 'stable_diffusion_uncertainty_guidance'
if not STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE.exists():
    STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE.mkdir()

STABLE_DIFFUSION_3_UNCERTAINTY_GUIDANCE = RESULTS / 'stable_diffusion_3_uncertainty_guidance'
if not STABLE_DIFFUSION_3_UNCERTAINTY_GUIDANCE.exists():
    STABLE_DIFFUSION_3_UNCERTAINTY_GUIDANCE.mkdir()

FLUX_UNCERTAINTY_GUIDANCE = RESULTS / 'flux_uncertainty_guidance'
if not FLUX_UNCERTAINTY_GUIDANCE.exists():
    FLUX_UNCERTAINTY_GUIDANCE.mkdir()

DIFFUSION_STARTING_POINTS = RESULTS / 'diffusion_starting_points'
if not DIFFUSION_STARTING_POINTS.exists():
    DIFFUSION_STARTING_POINTS.mkdir()

FID = RESULTS / 'fid'
if not FID.exists():
    FID.mkdir()

PLOT = RESULTS / 'plot'
if not PLOT.exists():
    PLOT.mkdir()

GENERATIONS = RESULTS / 'generations'
if not GENERATIONS.exists():
    GENERATIONS.mkdir()

DEBUG = RESULTS / 'debug'
if not DEBUG.exists():
    DEBUG.mkdir()

DATA = ROOT / 'data'
if not DATA.exists():
    DATA.mkdir()

IMAGENET = DATA / 'imagenet'
# if not IMAGENET.exists():
#     IMAGENET.mkdir()

IMAGENET_CLASS_MAP = IMAGENET / 'class_map.txt'

IMAGENET_VALIDATION_GROUND_TRUTH = IMAGENET / 'validation_ground_truth.txt'

IMAGENET_TRAIN = IMAGENET / 'train'
# if not IMAGENET_TRAIN.exists():
#     IMAGENET_TRAIN.mkdir()

IMAGENET_VAL = IMAGENET / 'val'
# if not IMAGENET_VAL.exists():
#     IMAGENET_VAL.mkdir()

IMAGENET_TEST = IMAGENET / 'test'
# if not IMAGENET_TEST.exists():
#     IMAGENET_TEST.mkdir()

IMAGENET64 = DATA / 'imagenet64'
# if not IMAGENET64.exists():
#     IMAGENET64.mkdir()

IMAGENET64_TRAIN = IMAGENET64 / 'train'
# if not IMAGENET64_TRAIN.exists():
#     IMAGENET64_TRAIN.mkdir()

IMAGENET64_VAL = IMAGENET64 / 'val'
# if not IMAGENET64_VAL.exists():
#     IMAGENET64_VAL.mkdir()

IMAGENET128 = DATA / 'imagenet128'
# if not IMAGENET128.exists():
#     IMAGENET128.mkdir()

IMAGENET128_TRAIN = IMAGENET128 / 'train'
# if not IMAGENET128_TRAIN.exists():
#     IMAGENET128_TRAIN.mkdir()

IMAGENET128_VAL = IMAGENET128 / 'val'
# if not IMAGENET128_VAL.exists():
#     IMAGENET128_VAL.mkdir()

IMAGENET128_TEST = IMAGENET128 / 'test'
# if not IMAGENET128_TEST.exists():
#     IMAGENET128_TEST.mkdir()

IMAGENET256 = DATA / 'imagenet256'
# if not IMAGENET256.exists():
#     IMAGENET256.mkdir()

IMAGENET256_TRAIN = IMAGENET256 / 'train'
# if not IMAGENET256_TRAIN.exists():
#     IMAGENET256_TRAIN.mkdir()

IMAGENET256_VAL = IMAGENET256 / 'val'
# if not IMAGENET256_VAL.exists():
#     IMAGENET256_VAL.mkdir()

IMAGENET256_TEST = IMAGENET256 / 'test'
# if not IMAGENET256_TEST.exists():
#     IMAGENET256_TEST.mkdir()

IMAGENET512 = DATA / 'imagenet512'

IMAGENET512_TRAIN = IMAGENET512 / 'train'
IMAGENET512_VAL = IMAGENET512 / 'val'
IMAGENET512_TEST = IMAGENET512 / 'test'

LSUN_CHURCHES256 = DATA / 'lsun-churches256'
LSUN_CHURCHES256_TRAIN = LSUN_CHURCHES256 / 'church_outdoor_train_lmdb'
LSUN_CHURCHES256_VAL = LSUN_CHURCHES256 / 'church_outdoor_val_lmdb'

CIFAR10 = DATA / 'cifar10'
CIFAR10_TRAIN = CIFAR10 / 'images' / 'train'
CIFAR10_TEST = CIFAR10 / 'images' / 'test'

DATASET_FID = RESULTS / 'dataset_fid'
if not DATASET_FID.exists():
    DATASET_FID.mkdir()


FINETUNING = RESULTS / 'finetuning'
if not FINETUNING.exists():
    FINETUNING.mkdir()


MODELS = ROOT / 'models'
if not MODELS.exists():
    MODELS.mkdir()

UVIT_IMAGENET64_CKPT = MODELS / 'imagenet64_uvit_mid.pth'

INTERMEDIATES = RESULTS / 'intermediates'

if not INTERMEDIATES.exists():
    INTERMEDIATES.mkdir()

SCORE_DATASET_FID = RESULTS / 'score_dataset_pytorch_fid'
if not SCORE_DATASET_FID.exists():
    SCORE_DATASET_FID.mkdir()

SCORE_UNCERTAINTY = RESULTS / 'score-uncertainty'
if not SCORE_UNCERTAINTY.exists():
    SCORE_UNCERTAINTY.mkdir()

THRESHOLD = RESULTS / 'thresholds'
if not THRESHOLD.exists():
    THRESHOLD.mkdir()

HOME = Path(os.environ['HOME'])
DATASETS = HOME / 'Datasets'

BPD = RESULTS / 'bpd'
if not BPD.exists():
    BPD.mkdir()

GLOBAL_IMAGENET = DATASETS / 'imagenet'
GLOBAL_IMAGENET64 = DATASETS / 'imagenet64'
GLOBAL_IMAGENET128 = DATASETS / 'imagenet128'
GLOBAL_IMAGENET256 = DATASETS / 'imagenet256'
GLOBAL_IMAGENET512 = DATASETS / 'imagenet512'
GLOBAL_CIFAR10 = DATASETS / 'cifar10'

def make_symlinks():
    if GLOBAL_IMAGENET.exists():
        if not IMAGENET.exists():
            GLOBAL_IMAGENET.symlink(IMAGENET)
    if GLOBAL_IMAGENET64.exists():
        if not IMAGENET64.exists():
            GLOBAL_IMAGENET64.symlink(IMAGENET64)
    if GLOBAL_IMAGENET128.exists():
        if not IMAGENET128.exists():
            GLOBAL_IMAGENET128.symlink(IMAGENET128)
    if GLOBAL_IMAGENET256.exists():
        if not IMAGENET256.exists():
            GLOBAL_IMAGENET256.symlink(IMAGENET256)
    if GLOBAL_IMAGENET512.exists():
        if not IMAGENET512.exists():
            GLOBAL_IMAGENET512.symlink(IMAGENET512)
    if GLOBAL_CIFAR10.exists():
        if not CIFAR10.exists():
            GLOBAL_CIFAR10.symlink(CIFAR10)
        
AUSE = RESULTS / 'ause'
if not AUSE.exists():
    AUSE.mkdir()

make_symlinks()