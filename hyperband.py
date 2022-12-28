# Imports
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torchvision
from torchvision import datasets
from torchvision import transforms

import numpy as np

##

DEVICE = torch.device("cpu" if torch.cuda.device_count() < 1 else "cuda:0")
batch_size = 4
CLASSES = 2
DIR = os.getcwd()
EPOCHS = 5
N_TRAIN_EXAMPLES = batch_size * 30
N_VALID_EXAMPLES = batch_size * 10

print("Device: {}\n\
        Batch size: {}\n\
        Classes: {}\n\
        Dir: {}\n\
        Epochs: {}\n\
        Number of training examples: {}\n\
        Number of validation examples: {}"\
        .format(DEVICE, batch_size, CLASSES, DIR, EPOCHS, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES))

##

model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

##

from torchvision import transforms as T
img_size = 224

transform = T.Compose([
                T.Resize((img_size,img_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

##

from torch.utils.data import DataLoader
from configs import Inputs
from utils.augmentations import get_transforms
from utils.data import FullRadiographSexDataset

val_dataset = FullRadiographSexDataset(root_dir=Inputs.DATASET_DIR,
                                       fold_nums=Inputs().val_folds,
                                       transforms=get_transforms(Inputs(), subset=["train"]))

val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

train_dataset = FullRadiographSexDataset(root_dir=Inputs.DATASET_DIR,
                                         fold_nums=Inputs().train_folds,
                                         transforms=get_transforms(Inputs(), subset=["train"]))

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)

##

config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "optimizer_name": tune.choice(["Adam", "RMSprop", "SGD"])
}

##

def objective(config):

    # Gerar o modelo
    #model = define_model(trial).to(DEVICE)
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(DEVICE)

    # Gerar optimizer
    optimizer_name = config['optimizer_name']
    lr = config["lr"]
    
    print("opt_name:", optimizer_name, "\nlr:", lr)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader, valid_loader = train_dataloader, val_dataloader

    criterion = F.nll_loss

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if epoch_steps % 5 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
            running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_dataloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        accuracy= correct / total

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss.cpu(), accuracy=accuracy)

    return accuracy

##

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=EPOCHS,
        grace_period=1,
        reduction_factor=2)

reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

result = tune.run(partial(objective),
                  resources_per_trial={"cpu": 2, "gpu": 1},
                  config=config,
                  num_samples=N_TRAIN_EXAMPLES,
                  scheduler=scheduler,
                  progress_reporter=reporter,
)

##