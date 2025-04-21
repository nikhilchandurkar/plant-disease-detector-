# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     # Hyperparameters
#     batch_size = 64
#     num_epochs = 30
#     learning_rate = 0.001

#     # Data transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Datasets
#     train_data = datasets.ImageFolder("data/PlantVillage/train", transform=transform)
#     val_data = datasets.ImageFolder("data/PlantVillage/val", transform=transform)

#     # Data loaders
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     # Load pre-trained ResNet-50
#     model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#     num_classes = len(train_data.classes)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
#         loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
#         for images, labels in loop:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
#         print(f"Epoch [{epoch+1}/{num_epochs}] => Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
#     torch.save(model, "plant_disease_model.pth")
#     print("✅ Model saved to 'plant_disease_model.pth'")

# if __name__ == "__main__":
#     main()




















import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess
import time

def print_gpu_temp():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        )
        temp = result.decode("utf-8").strip()
        print(f"🌡️  GPU Temperature: {temp}°C")
    except Exception as e:
        print("Could not retrieve GPU temperature:", e)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)
    print_gpu_temp()

    # Hyperparameters
    batch_size = 48  # slightly lower to reduce heat
    num_epochs = 30
    learning_rate = 0.001

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder("data/PlantVillage/train", transform=transform)
    val_data = datasets.ImageFolder("data/PlantVillage/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}] => Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        print_gpu_temp()
        time.sleep(3)  

    torch.save(model, "plant_disease_model.pth")
    print("✅ Model saved to 'plant_disease_model.pth'")

if __name__ == "__main__":
    main()
