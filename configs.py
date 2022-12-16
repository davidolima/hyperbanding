import torch.nn as nn
import torch.optim as optim

class Inputs:
    # WandB
    # PROJECT = 'sex-estimation'
    # ENTITY = 'ivision-radiographs'

    # General configs
    DATASET_DIR: str = '/home/david/Documents/iVision/data/pan-radiographs/'

    SEED = 7 # Unused until now
    NUM_WORKERS: int = 2
    MEAN = [0.485, 0.456, 0.406]
    STDV = [0.229, 0.224, 0.225]

    # TODO: check a way of putting the criterion as an input argument
    #       and the better reduction
    CRITERION = nn.CrossEntropyLoss()
    # TODO: check a way of putting the optimizer as an input argument
    OPTIMIZER = optim.SGD

    # Base hyperparameters
    EPOCHS: int = 120
    WARMUP_EPOCHS: int = 5

    # Base learning rates
    BASE_LR = 3e-4
    VAL_BATCH_MULTIPLIER: int = 4
    AUGMENTATION_NAME = None

    MODEL_CONFIGS = {
        'resnet-50': {
            'name': 'resnet-50',
            'img_size': 224,
            'batch_size': 64
        },
        'efficientnet-b0': {
            'name': 'efficientnet-b0',
            'img_size': 224,
            'batch_size': 64
            # 'base_lr': 1.2e-4
        },
        'efficientnet-b1': {
            'name': 'efficientnet-b1',
            'img_size': 240,
            'batch_size': 128
        },
        'efficientnet-b2': {
            'name': 'efficientnet-b2',
            'img_size': 260,
            'batch_size': 128
        },
        'efficientnet-b3': {
            'name': 'efficientnet-b3',
            'img_size': 300,
            'batch_size': 128
        },
        'efficientnet-b4': {
            'name': 'efficientnet-b4',
            'img_size': 380,
            'batch_size': 16
        },
        # TODO: check the batch sizes
        'efficientnet-b5': {
            'name': 'efficientnet-b5',
            'img_size': 456,
            'batch_size': 128
        },
        'efficientnet-b6': {
            'name': 'efficientnet-b6',
            'img_size': 528,
            'batch_size': 128
        },
        'efficientnet-b7': {
            'name': 'efficientnet-b7',
            'img_size': 600,
            'batch_size': 8
        },
        'efficientnet_v2_s': {
            'name': 'efficientnet_v2_s',
            'img_size': 480,
            'batch_size': 16
        },
        'efficientnet_v2_m': {
            'name': 'efficientnet_v2_m',
            'img_size': 480,
            'batch_size': 8
        },
        'efficientnet_v2_l': {
            'name': 'efficientnet_v2_l',
            'img_size': 480,
            'batch_size': 4
        },
        'efficientnet_v2_xl': {
            'name': 'efficientnet_v2_xl',
            'img_size': 480,
            'batch_size': 2
        }
    }

    def __init__(
        self,
        selected_model: str = 'efficientnet_v2_l',
        criterion_name: str = 'crossentropy',
        optimizer_name: str = 'sgd',
        num_folds: int = 1,
        val_fold: int = 5,
        train_folds: int = [1, 2, 3],
        val_folds: int = [4, 5],
        crop_side: str='right',
        remove_border: int=75,
        dataset='full',
        name: str = 'efficientnet_v2_l-fold-4'
    ):
        # TODO: allow different criterions and optimizers
        if criterion_name != 'crossentropy':
            raise ValueError(f'Criterion {criterion_name} not supported yet.')

        if optimizer_name != 'sgd':
            raise ValueError(f'Optimizer {optimizer_name} not supported yet.')

        self.model_config = Inputs.MODEL_CONFIGS[selected_model]
        self.img_size = self.model_config['img_size']

        # Architecture
        self.model_name = selected_model
        self.name = name #f'sex-estimation-{self.model_name}'

        self.batches_and_lr = self.init_batch_sizes_and_lr()
        self.lr = self.batches_and_lr['lr']

        self.num_folds = num_folds
        self.val_fold = val_fold

        self.train_folds = train_folds
        self.val_folds = val_folds

        self.crop_side = crop_side
        self.remove_border = remove_border

        self.dataset = dataset

        # self.wandb = {
        #     'name': self.name,
        #     'model_name': self.model_name,
        #     'dataset_dir': Inputs.DATASET_DIR,
        #     'epochs': Inputs.EPOCHS,
        #     'warmup_epochs': Inputs.WARMUP_EPOCHS,
        #     'learning_rate': self.lr,
        #     'train_batch_size': self.batches_and_lr['train_batch_size'],
        #     'val_batch_size': self.batches_and_lr['val_batch_size'],
        #     'image_size': self.img_size,
        #     'criterion': criterion_name,
        #     'optimizer': optimizer_name,
        #     'num_folds': self.num_folds,
        #     'val_fold': self.val_fold,
        #     'train_folds': train_folds,
        #     'val_folds': val_folds,
        #     'crop_side': self.crop_side,
        #     'remove_border': self.remove_border
        #}

    def init_batch_sizes_and_lr(self):
        batches_and_lr = {}

        # batch sizes
        batches_and_lr['train_batch_size'] = self.model_config['batch_size']
        batches_and_lr['val_batch_size']   = \
                Inputs.VAL_BATCH_MULTIPLIER * self.model_config['batch_size']

        # learning rate
        batches_and_lr['lr'] = Inputs.BASE_LR * self.model_config['batch_size']

        return batches_and_lr