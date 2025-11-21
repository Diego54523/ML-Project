from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch
import numpy as np
from torchvision import models
from tqdm import tqdm

import os

dataSetPath = "/home/manu/Documents/Machine Learning/ML-Project/archive/MRI"
# dataSetPath = os.getenv("DATA_PATH")

# Preprocess
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=dataSetPath, transform=transformer)

batchSize = 32
shuffle = True
numWorkers = 4
divider = 0.8

trainSize = int(divider * len(dataset))
valSize = len(dataset) - trainSize

trainDataset, valDataset = random_split(
    dataset, 
    [trainSize, valSize],
    generator=torch.Generator().manual_seed(42)
    )

trainLoader = DataLoader(
    trainDataset, 
    batch_size=batchSize, 
    shuffle=shuffle, 
    num_workers=numWorkers
    )

valLoader = DataLoader(
    valDataset, 
    batch_size=batchSize, 
    shuffle=shuffle, 
    num_workers=numWorkers
    )

# print("Training set size: {}".format(len(trainDataset)))
# print("Validation set size: {}".format(len(valDataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print(model)

featureExtractor = torch.nn.Sequential(
    *list(model.children())[:-1]).to(device).eval()

# print("Feature Extractor:\n", featureExtractor)

batch = next(iter(trainLoader))
imgs, labels = batch 

with torch.no_grad():
    feats = featureExtractor(imgs.to(device))
    feats = feats.view(feats.size(0), -1)

# print("Image shape: ", imgs.shape)
# print("Feature shape: ", feats.shape)

def extract_embeddings(dataloader, feature_extractor, device):
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, labels = batch
            imgs = imgs.to(device)

            feats = feature_extractor(imgs)
            feats = feats.view(feats.size(0), -1)

            all_feats.append(feats.cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_feats.numpy(), all_labels.numpy()

train_feats, train_labels = extract_embeddings(
    trainLoader, featureExtractor, device)

te_feats, te_labels = extract_embeddings(
    valLoader, featureExtractor, device)

print("Train features shape: ", train_feats.shape)
print("Train labels shape: ", train_labels.shape)
print("Validation features shape: ", te_feats.shape)
print("Validation labels shape: ", te_labels.shape)

classes = dataset.classes
print("Classes: ", classes)

np.savez_compressed("mri_features.npz",
    train_feats=train_feats,
    train_labels=train_labels,
    val_feats=te_feats,
    val_labels=te_labels,
    classes=np.array(classes)
)

data = np.load("mri_features.npz")
X_train = data['train_feats']
y_train = data['train_labels']
X_val = data['val_feats']
y_val = data['val_labels']

classes = data['classes'].tolist()

print("Train features loaded: ", X_train.shape)
print("Train labels loaded: ", y_train.shape)
print("Validation features loaded: ", X_val.shape)
print("Validation labels loaded: ", y_val.shape)
print("Loaded classes: ", classes)

# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)    

y_pred = logreg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
