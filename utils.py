"""Utility functions utilized within the Milestone 2 NHATs data project."""

import ast
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torchvision
import torchvision.models as models
import torch.nn as nn
from proj_models import ConvNet


def open_dict(path):
    """Take in path and opens file."""
    cust_file = open(path, "r")
    contents = cust_file.read()
    dictionary = ast.literal_eval(contents)
    cust_file.close()

    return dictionary


def collate_fn(batch):
    """From pytorch - way to bypass corrupt or non-existent data."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def set_model(m, model_ext, device):
    """Determine which model is to be used in the notebook."""
    if torch.cuda.is_available():
        print("First Model training on GPU")

        if m == "First model":
            # Create model object
            model = ConvNet()
            model = model.to(device)  # (float).cuda()

        elif m == "pre-trained":
            mPATH = "/content/gdrive/MyDrive/Colab Notebooks/Models/cnn_512_662.model{}".format(
                model_ext
            )
            model = ConvNet()
            model.load_state_dict(torch.load(mPATH))
            model.to(device)
            print("New Model{} training on GPU".format(model_ext))

        elif m == "resnet":
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 6)
            model = model.to(device)
            print("RESNET Model training on GPU")

        elif m == "pre-trained-res":
            mPATH = "/content/gdrive/MyDrive/Colab Notebooks/Models/cnn_512_662.model{}".format(
                model_ext
            )
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 6)
            model.load_state_dict(torch.load(mPATH))
            model.to(device)
            print("New Model{} training on GPU".format(model_ext))

    else:
        print("CUDA is not available. Turn on GPU")

    return model


def accuracy(y_pred, y_test):
    """Calculate the accuracy of the model."""
    # Calculating model accuracy at each epoch
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_prob = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_prob == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)

    return acc


def train_val_model(
    epochs, model, train_loader, device, optimizer, criterion, validate_loader
):
    """Train model while evaluating with evalutaion hold out."""
    for epoch in range(1, epochs + 1):

        train_epoch_loss = 0
        train_epoch_acc = 0

        # set model in training mode
        model.train()
        print("\nEpoch$ : %d" % epoch)
        for x_train_batch, y_train_batch in tqdm(train_loader):
            x_train_batch = x_train_batch.to(
                device
            )  # (float).to(device) # for GPU support
            y_train_batch = y_train_batch.to(device)

            # sets gradients to 0 to prevent interference with previous epoch
            optimizer.zero_grad()

            # Forward pass through NN
            y_train_pred = model(x_train_batch)  # .to(float)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = accuracy(y_train_pred, y_train_batch)

            # Backward pass, updating weights
            train_loss.backward()
            optimizer.step()

            # Statistics
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.set_grad_enabled(False):
            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for x_val_batch, y_val_batch in tqdm(validate_loader):

                x_val_batch = x_val_batch.to(device)  # .to(float)
                y_val_batch = y_val_batch.to(device)

                # Forward pass
                y_val_pred = model(x_val_batch)  # .to(float)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = accuracy(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        # Prevent plateauing validation loss
        # scheduler.step(val_epoch_loss/len(validate_loader))

        accuracy_stats = {"train": [], "val": []}

        loss_stats = {"train": [], "val": []}

        loss_stats["train"].append(train_epoch_loss / len(train_loader))
        loss_stats["val"].append(val_epoch_loss / len(validate_loader))
        accuracy_stats["train"].append(train_epoch_acc / len(train_loader))
        accuracy_stats["val"].append(val_epoch_acc / len(validate_loader))

        print(
            f"Epoch {epoch+0:03}: Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(validate_loader):.5f}"
        )
        print(
            f"Train Acc: {train_epoch_acc/len(train_loader):.3f} | Val Acc: {val_epoch_acc/len(validate_loader):.3f}"
        )


def multiclass_roc_auc_score(y_test1, all_pred1, average="macro"):
    """Plot multiclass AUC score."""
    lb = LabelBinarizer()
    lb.fit(y_test1)
    y_test1 = lb.transform(y_test1)
    all_pred1 = lb.transform(all_pred1)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test1[:, idx].astype(int), all_pred1[:, idx])
        c_ax.plot(fpr, tpr, label="%s (AUC:%0.2f)" % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, "b-", label="Random Guessing")
    return roc_auc_score(y_test1, all_pred1, average=average)


def visualize_model(model, num_images=6):
    """Visualize the model."""
    was_training = model.training
    class_names = [0, 1, 2]
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validate_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title("predicted: {}".format(class_names[preds[j]]))
                imshow(
                    inputs.cpu().data[j].squeeze().permute(2, 1)
                )  # image1.squeeze().permute(1,2,0)
                # imshow(np.transpose(inputs.cpu().data[j], (2, 0, 1)))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
