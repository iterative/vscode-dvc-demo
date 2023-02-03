"""Model training and evaluation."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from dvclive import Live
from PIL import Image
from ruamel.yaml import YAML


def transform(dataset):
    """Get inputs and targets from dataset."""
    x = dataset.data.reshape(len(dataset.data), 1, 28, 28)/255
    y = dataset.targets
    return x, y


def train(model, x, y, lr, weight_decay):
    """Train a single epoch."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def predict(model, x):
    """Get model prediction scores."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
    return y_pred


def get_metrics(y, y_pred, y_pred_label):
    """Get loss and accuracy metrics."""
    metrics = {}
    criterion = torch.nn.CrossEntropyLoss()
    metrics["loss"] = criterion(y_pred, y).item()
    metrics["acc"] = (y_pred_label == y).sum().item()/len(y)
    return metrics


def evaluate(model, x, y):
    """Evaluate model and save metrics."""
    scores = predict(model, x)
    _, labels = torch.max(scores, 1)
    actual = [int(v) for v in y]
    predicted = [int(v) for v in labels]

    metrics = get_metrics(y, scores, labels)

    return metrics, actual, predicted

def get_confusion_image(actual, predicted, dataset):
    confusion = {}
    for n, (a, p) in enumerate(zip(actual, predicted)):
        image = np.array(dataset[n][0]) / 255
        confusion[(a, p)] = image

    max_i, max_j = 0, 0
    for (i, j) in confusion:
        if i > max_i:
            max_i = i
        if j > max_j:
            max_j = j

    frame_size = 30
    image_shape = (28, 28)
    incorrect_color = np.array((255, 100, 100), dtype="uint8")
    label_color = np.array((100, 100, 240), dtype="uint8")

    out_matrix = np.ones(shape=((max_i+2) * frame_size, (max_j+2) * frame_size, 3), dtype="uint8") * 240

    for i in range(max_i+1):
        if (i, i) in confusion:
            image = confusion[(i, i)]
            xs = (i + 1) * frame_size + 1
            xe = (i + 2) * frame_size - 1
            ys = 1
            ye = frame_size - 1
            for c in range(3):
                out_matrix[xs:xe, ys:ye, c] = (1 - image) * label_color[c]
                out_matrix[ys:ye, xs:xe, c] = (1 - image) * label_color[c]

    for (i, j) in confusion:
        image = confusion[(i, j)]
        assert image.shape == image_shape
        xs = (i + 1) * frame_size + 1
        xe = (i + 2) * frame_size - 1
        ys = (j + 1) * frame_size + 1
        ye = (j + 2) * frame_size - 1
        assert (xe-xs, ye-ys) == image_shape
        if i != j:
            for c in range(3):
                out_matrix[xs:xe, ys:ye, c] = (1 - image) * incorrect_color[c]

    return out_matrix


def save_hist(predicted, fname):
    """Save histogram of predictions."""
    unique, counts = np.unique(predicted, return_counts=True)
    hist = np.asarray((unique, counts)).T
    np.savetxt(fname, hist, delimiter=",", fmt="%i", header="digit,preds",
               comments="")


def main():
    """Train model and evaluate on test data."""
    torch.manual_seed(473987)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 10),
    )
    live = Live(dir="training",dvcyaml=False, report=None)

    # Load model.
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt"))

    # Load params.
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)

    # Load train and test data.
    mnist_train = torchvision.datasets.MNIST("data", download=True)
    x_train, y_train = transform(mnist_train)
    mnist_test = torchvision.datasets.MNIST("data", download=True, train=False)
    x_test, y_test = transform(mnist_test)

    # Iterate over training epochs.
    for epoch in range(params["epochs"]):
        print(f"EPOCH: {epoch + 1} / {params['epochs']}")
        train(model, x_train, y_train, params["lr"], params["weight_decay"])
        torch.save(model.state_dict(), "model.pt")
        # Evaluate and checkpoint.
        metrics_train, _, _ = evaluate(model, x_train, y_train)
        for k, v in metrics_train.items():
            live.log_metric(f"train/{k}", v)
        metrics_test, actual, predicted = evaluate(model, x_test, y_test)
        for k, v in metrics_test.items():
            live.log_metric(f"test/{k}", v)
        unique, counts = np.unique(predicted, return_counts=True)
        missclassified = get_confusion_image(actual, predicted, mnist_test)
        save_hist(predicted, "hist.csv")
        live.log_image("misclassified.jpg", missclassified)
        live.log_sklearn_plot("confusion_matrix", actual, predicted)
        live.next_step()


if __name__ == "__main__":
    main()
