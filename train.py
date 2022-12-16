"""
Main module for training the networks for gender estimation.
Author: Bernardo Silva (https://github.com/bpmsilva)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb

from configs import Inputs
from utils.augmentations import get_transforms
from utils.data import RadiographSexDataset, FullRadiographSexDataset
from utils.lambda_schedulers import linear_warmup

# def initialize_wandb(inputs):
#     wandb.init(name=inputs.name, project=inputs.PROJECT, entity=inputs.ENTITY)
#     wandb.config = inputs.wandb

def get_classification_model(model_name, num_classes):
    if model_name == 'resnet-50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet-b0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet-b4':
        model = torchvision.models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet-b7':
        model = torchvision.models.efficientnet_b7(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = torchvision.models.efficientnet_v2_s(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_v2_m':
        model = torchvision.models.efficientnet_v2_m(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_v2_l':
        model = torchvision.models.efficientnet_v2_l(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_v2_xl':
        model = torchvision.models.efficientnet_v2_xl(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise Exception(f'Model {model_name} is not supported')
    return model

def load_checkpoint(checkpoint_path, model, optimizer, device):
    # TODO: add and check device
    checkpoint = torch.load(checkpoint_path)

    # load variables
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']

    return step

def compute_metrics(outputs, labels):
    # convert outputs to the predicted classes
    _, pred = torch.max(outputs, 1)

    # compare predictions to true label
    total = len(labels)
    true_positives = (pred & labels.data.view_as(pred)).sum().item()
    true_negatives = ((1 - pred) & (1 - labels).data.view_as(pred)).sum().item()
    false_positives = (pred & (1 - labels).data.view_as(pred)).sum().item()
    false_negatives = ((1 - pred) & labels.data.view_as(pred)).sum().item()

    return {
        'tp': true_positives,
        'tn': true_negatives,
        'fp': false_positives,
        'fn': false_negatives,
        # 'accuracy': accuracy,
        # 'precision': true_positives / (true_positives + false_positives),
        # 'recall': true_positives / (true_positives + false_negatives),
        # 'f1': 2 * true_positives / (2 * true_positives + false_positives + false_negatives),
        'total': total
    }

def train_step(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    device
):
    running_loss, total = 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for imgs, labels in train_loader:
        # put model in training mode
        model.train()
        # send images and labels to device
        imgs, labels = imgs.to(device), labels.to(device)

        # feedforward and loss with mixed-precision
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # TODO: check if this output is logits, probabilities or log of probabilities
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # sum up the loss
        running_loss += loss.item() * len(imgs)

        # backpropagation with mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        metrics = compute_metrics(outputs, labels)
        tp += metrics['tp']
        tn += metrics['tn']
        fp += metrics['fp']
        fn += metrics['fn']
        total += metrics['total']

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    #print(f'Train accuracy: {accuracy:.4f}')
    print(f'Train precision: {precision:.4f}')
    print(f'Train recall: {recall:.4f}')
    print(f'Train F1: {f1:.4f}')

    print(f'Training loss: {running_loss / len(train_loader):.5f}')
    print(f'Training accuracy: {100*accuracy:.2f} (%)')
    
    # wandb log
    # wandb.log({
    #     'train_loss': running_loss / len(train_loader),
    #     'train_accuracy': accuracy

def val_step(
    model,
    val_loader,
    criterion,
    device
):
    with torch.no_grad():
        running_loss, total = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for imgs, labels in val_loader:
            # put model in evaluation mode
            model.eval()
            # send images and labels to device
            imgs, labels = imgs.to(device), labels.to(device)

            # feedforward
            # TODO: check if I should add mixed-precision here
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # sum up the loss
            running_loss += loss.item() * len(imgs)

            metrics_dict = compute_metrics(outputs, labels)
            tp += metrics_dict['tp']
            tn += metrics_dict['tn']
            fp += metrics_dict['fp']
            fn += metrics_dict['fn']
            total += metrics_dict['total']

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)

        print(f'Validation loss: {running_loss / len(val_loader):.5f}')
        print(f'Validation accuracy: {100*accuracy:.2f} (%)')

        # wandb log
        val_loss = running_loss / len(val_loader)
        # wandb.log({
        #     'val_loss': val_loss,
        #     'val_accuracy': accuracy
        # })

        #print(f'Validation accuracy: {accuracy:.4f}')
        print(f'Validation precision: {precision:.4f}')
        print(f'Validation recall: {recall:.4f}')
        print(f'Validation F1: {f1:.4f}')

        return {
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def train(
    name,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs,
    save_model=True,
    load_model=None
):
    # not that important, but apparently gives performance boost
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')

    if load_model:
        step = load_checkpoint(load_model, model, optimizer, device)
    else:
        step = 0

    # for more than 1 GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        model = nn.DataParallel(model)
    else:
        print('Using a single GPU\n')
    model.to(device)

    # mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # training loop
    max_val_accuracy, max_val_f1, min_val_loss = 0, 0, 10000000000
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')

        train_step(model, train_loader, optimizer, scheduler, criterion, scaler, device)
        print()
        val_metrics = val_step(model, valid_loader, criterion, device)
        val_accuracy, val_f1, val_loss = val_metrics['accuracy'], val_metrics['f1'], val_metrics['loss']
        print()

        if max_val_accuracy < val_accuracy:
            print(f'Accuracy increased from {max_val_accuracy:.4f}' + \
                  f' to {val_accuracy:.4f} ({epoch}/{epochs})')

            max_val_accuracy = val_accuracy
            if save_model:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint, filename=f'checkpoint-{name}-max-acc.pth.tar')

        if max_val_f1 < val_f1:
            print(f'F1 increased from {max_val_f1:.4f}' + \
                  f' to {val_f1:.4f} ({epoch}/{epochs})')

            max_val_f1 = val_f1
            if save_model:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint, filename=f'checkpoint-{name}-max-f1.pth.tar')

        if min_val_loss > val_loss:
            print(f'Validation loss decreased from {min_val_loss:.2f}' + \
                  f' to {val_loss:.2f} ({epoch}/{epochs})')

            min_val_loss = val_loss
            if save_model:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint, filename=f'checkpoint-{name}-min-val-loss.pth.tar')

def main():
    inputs = Inputs()

    # initialize_wandb(inputs)

    folds = {
        'train': [fold for fold in range(1, inputs.num_folds + 1) if fold != inputs.val_fold],
        'val':   [inputs.val_fold]
    }

    # get the datasets
    datasets, dataloaders = dict(), dict()
    for subset in ['train', 'val']:
        if inputs.dataset == 'full':
            datasets[subset] = FullRadiographSexDataset(
                root_dir=inputs.DATASET_DIR,
                fold_nums=inputs.train_folds if subset == 'train' else inputs.val_folds,
                transforms=get_transforms(inputs, subset=subset)
            )
        else:
            datasets[subset] = RadiographSexDataset(
                root_dir=inputs.DATASET_DIR,
                fold_nums=folds[subset],
                transforms=get_transforms(inputs, subset=subset),
                crop_side=inputs.crop_side,
                border=inputs.remove_border
            )

        dataloaders[subset] = DataLoader(
            datasets[subset],
            batch_size=inputs.batches_and_lr[f'{subset}_batch_size'],
            shuffle=subset=='train',
            num_workers=inputs.NUM_WORKERS
        )

    #print(inputs)
    print(f'\nModel is {inputs.model_name}')
    print(f'\nVal fold is {inputs.val_folds}')
    print()
    model = get_classification_model(inputs.model_name, 2)

    # optimizer and scheduler
    optimizer = inputs.OPTIMIZER(model.parameters(), lr=inputs.lr)
    warmup_steps = len(dataloaders['train']) * inputs.WARMUP_EPOCHS
    scheduler = linear_warmup(optimizer, warmup_steps)

    train(
        name=inputs.name,
        model=model,
        criterion=inputs.CRITERION,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dataloaders['train'],
        valid_loader=dataloaders['val'],
        epochs=inputs.EPOCHS,
    )

if __name__ == '__main__':
    main()