import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
from helper_TMLFN import accuracy_fn, print_train_time, train_step, test_step, eval_model, make_predictions, plot_model_predictions, p_confusion_matrix
from PIL import Image
from pathlib import Path
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.MNIST(root="./data/MNIST",
                                train=True, 
                                download=True, 
                                transform=ToTensor(),
                                target_transform=None)

test_dataset = datasets.MNIST(root="./data.MNIST",
                                train=False, 
                                download=True, 
                                transform=ToTensor(),
                                target_transform=None)

BATCH_SIZE = 32
class_names = train_dataset.classes

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_features_batch, train_labels_batch = next(iter(train_dataloader))

class nnNumModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_1 = nnNumModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

train_model = False
show_model_quality = False
if train_model:
    train_start_timer = timer()

    for epoch in tqdm(range(20)):
        print(f"Epoch {epoch}\n------------------------")
        train_step(model=model_1, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
        test_step(model=model_1, test_data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

    train_end_timer = timer()

    total_train_time = print_train_time(train_start_timer, train_end_timer, device)
    model_result = eval_model(model=model_1, data_loader=train_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    print(model_result)

    MODEL_PATH = Path("./models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "numModel_1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model_1.state_dict(), MODEL_SAVE_PATH)

else:
    loaded_model_1 = nnNumModel(input_shape=1, hidden_units=10, output_shape=len(class_names))
    loaded_model_1.load_state_dict(torch.load("./models/numModel_1.pth"))
    loaded_model_1.to(device)

    if show_model_quality:
        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(test_dataset), k=9):
            test_samples.append(sample)
            test_labels.append(label)

        pred_probs = make_predictions(model=loaded_model_1, data=test_samples, device=device)
        pred_classes = pred_probs.argmax(dim=1)

        plot_model_predictions(pred_probs=pred_probs, pred_classes=pred_classes, test_labels=test_labels, test_samples=test_samples, class_names=class_names)
        p_confusion_matrix(model=loaded_model_1, test_data_loader=test_dataloader, data=test_dataset, class_names=class_names, device=device)

    else:
        img = Image.open("./test_img/img_1.jpg")
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        model_choice = loaded_model_1(img_tensor).argmax(dim=1)
        print(f"The model's choice: {model_choice.item()}")


