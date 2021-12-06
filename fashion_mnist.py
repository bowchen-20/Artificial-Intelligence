############################################################
# CIS 521: Neural Network for Fashion MNIST Dataset
############################################################

student_name = "Bowen Chen"

############################################################
# Imports
############################################################

import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools


# Include your imports here, if any are used.


############################################################
# Neural Networks
############################################################

def load_data(file_path, reshape_images):

    with open(file_path) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        next(csv_reader)

        labels = []

        pixels = []

        for i in csv_reader:

            labels.append(int(i[0]))

            if reshape_images:

                output_array = np.array([int(j) for j in i[1:]]).reshape(1, 28, 28)

            else:

                output_array = [int(j) for j in i[1:]]

            pixels.append(output_array)

    return np.array(pixels), np.array(labels)


# PART 2.2
class EasyModel(torch.nn.Module):
    def __init__(self):

        super(EasyModel, self).__init__()

        self.fc = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):

        x = self.fc(x)

        return x


# PART 2.3
class MediumModel(torch.nn.Module):

    def __init__(self):

        super(MediumModel, self).__init__()

        self.fc1 = torch.nn.Linear(28 * 28, 400)

        self.fc2 = torch.nn.Linear(400, 400)

        self.fc3 = torch.nn.Linear(400, 10)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x)


# PART 2.4
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform(self.cnn1.weight)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform(self.cnn2.weight)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 1024)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(1024, 256)
        self.fcrelu1 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        out = self.fcrelu1(out)

        out = self.fc3(out)
        return out

############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100. * accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100. * f1_score(y_true, y_predicted, average="weighted"):.4f}')


def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted


def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(f'Easy Model: '
          f'Final Train Accuracy: {100. * accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100. * f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(f'Medium Model: '
          f'Final Train Accuracy: {100. * accuracy_score(y_true_medium, y_pred_medium):.4f},',
          f'Final F1 Score: {100. * f1_score(y_true_medium, y_pred_medium, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(f'Advanced Model: '
          f'Final Train Accuracy: {100. * accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100. * f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')


############################################################
# Feedback
############################################################

feedback_question_1 = """
Shirt and pullover are the two classes that the easy model confused the most
Shirt and T-shirt are the two classes that the medium model confused the most
Coat and pullover are the two classes that the advanced model confused the most
"""

feedback_question_2 = """
Used two CNNs in combination with one max pulling and one average pooling, followed
by three fully connected layer at the end, accuracy is about 93%
"""

feedback_question_3 = 30

feedback_question_4 = """
I had some trouble understanding what does neural networks do in the very beginning, 
and I wasn't sure how to tune in and out exactly and how what effect kernel size and 
padding size would have on the final result
resources consulted: https://github.com/zalandoresearch/fashion-mnist
https://github.com/Abhi-H/CNN-with-Fashion-MNIST-dataset/blob/master/FashionMNISTwithCNN.py
https://github.com/meghanabhange/FashionMNIST-3-Layer-CNN/blob/master/Fashion.py
http://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
"""

feedback_question_5 = """
The assignment was pretty fun in general as we get to tune with the parameters ourselves. 
In some cases its a bit unclear whether certain parameters have a hiddren relationship
"""

if __name__ == '__main__':
    main()

# class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# num_epochs = 2
# batch_size = 100
# learning_rate = 0.001
# data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset('dataset.csv', True), batch_size=batch_size, shuffle=True)
# advanced_model = AdvancedModel()
# train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
# y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
#
# print(f'Advanced Model: '
#     f'Final Train Accuracy: {100.* accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
#     f'Final F1 Score: {100.* f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
# plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')