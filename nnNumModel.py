import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from PIL import Image
from pathlib import Path
import helper_TMLFN as helper_fn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.MNIST(root="./data",
                                split="balanced",
                                train=True, 
                                download=True, 
                                transform=transforms.Compose([
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    ToTensor()
                                ]),
                                target_transform=None)

test_dataset = datasets.MNIST(root="./data",
                                split="balanced",
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    ToTensor()
                                ]),
                                target_transform=None)

BATCH_SIZE = 32
class_names = train_dataset.classes

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class nn_NL_Model(nn.Module):
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
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model_1 = nn_NL_Model(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

epochs = 25
train = False
show_model_quality = True
if train:
    start_training = timer()
    model_results = helper_fn.train_model(model=model_1, 
                                            train_dataloader=train_dataloader, 
                                            test_dataloader=test_dataloader, 
                                            loss_fn=loss_fn, 
                                            optimizer=optimizer,
                                            epochs=epochs, 
                                            device=device)

    end_training = timer()
    total_train_time = helper_fn.print_train_time(start=start_training, end=end_training, device=device)
    helper_fn.plot_loss_curves(model_results, epochs)

    MODEL_PATH = Path("./models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "nl_model_1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model_1.state_dict(), MODEL_SAVE_PATH)

else:
    loaded_model_1 = nn_NL_Model(input_shape=1, hidden_units=10, output_shape=len(class_names))
    loaded_model_1.load_state_dict(torch.load("./models/nl_model_1.pth"))
    loaded_model_1.to(device)

    if show_model_quality:
        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(test_dataset), k=9):
            test_samples.append(sample)
            test_labels.append(label)

        pred_probs = helper_fn.make_predictions(model=loaded_model_1, data=test_samples, device=device)
        pred_classes = pred_probs.argmax(dim=1)

        helper_fn.plot_model_predictions(pred_probs=pred_probs, pred_classes=pred_classes, test_labels=test_labels, test_samples=test_samples, class_names=class_names)
        helper_fn.p_confusion_matrix(model=loaded_model_1, test_data_loader=test_dataloader, data=test_dataset, class_names=class_names, device=device)

    else:
        img = Image.open("./test_img/img_1.jpg")
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        model_choice = loaded_model_1(img_tensor).argmax(dim=1)
        print(f"The model's choice: {class_names[model_choice.item()]}")

