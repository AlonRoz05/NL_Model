import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_true)) * 100
    return accuracy

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_accuracy": acc}

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device):
    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_accuracy += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_data_loader)
        test_accuracy /= len(test_data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}%\n")

def make_predictions(model: torch.nn.Module, data: list, device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

def plot_model_predictions(pred_probs, pred_classes, test_labels, test_samples, class_names):
    plt.figure(figsize=(9,9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")

        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]

        title_text = f"Pred {pred_label} | Truth {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

        plt.show()


def p_confusion_matrix(model: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader, data, class_names, device):
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_data_loader, desc="Making predictions..."):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=y_pred_tensor, target=data.targets)

    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10,7))
    plt.show()


