import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from typing import List

def accuracy_fn(y_true, y_pred):
    """Returns the accuracy of the given y_true and y_pred values"""
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_true)) * 100
    return accuracy

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    """Evaluates the model on a given training data loader and loss function and return the results"""
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

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device):
    """Trains a model on a given data loader"""
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss  
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device):
    """Tests how good the model is trained"""
    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y).item()
            test_accuracy += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)
        return test_loss, test_accuracy
    
def train_model(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, device):
    """Training and testing the model and returns the results of the training and testing loos and accuracy to plot loss and accuracy curves."""
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
        test_loss, test_acc = test_step(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
        
        print(f"Epoch: {epoch}\n--------------------------------")
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2f}\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}\n--------------------------------")
        
        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def make_predictions(model: torch.nn.Module, data: list, device):
    """doing predictions on 9 images from dataset"""
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
    """Plots the predictions from test_samples"""
    plt.figure(figsize=(9,9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")
        plt.axis(False)

        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]

        title_text = f"Pred {pred_label} | Truth {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

    plt.show()

def p_confusion_matrix(model: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader, data, class_names, device):
    """Plots confusion matrix"""
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

def plot_loss_curves(results: dict[str, List[float]], epochs: int):
    """Plots Training curves for results dictionary"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

