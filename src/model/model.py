import torch
from torch import nn
from collections import OrderedDict
from tqdm import tqdm
from time import time
import numpy as np
from ..config import BATCH_SIZE, EPOCHS, INITIAL_LEARNING_RATE, CLASSES_AMOUNT
# For training
from torch import optim
# For testing
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


class Model:
    """
    A wrapper for a pytorch model, with an easier to use interface.
    """

    # Initialize the device to use (cuda for GPU, else CPU).
    # Prefer to use the GPU if available, because it is faster.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # Create the model structure
        self._model = nn.Sequential(OrderedDict([
            # Convolution part
            ("conv1", nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 5))),
            ("relu1", nn.ReLU()),
            ("max_pool1", nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))),

            ("flatten", nn.Flatten(1)),

            # Fully connected part
            ("fc1", nn.Linear(in_features=2450, out_features=800)),
            ("relu2", nn.ReLU()),
            ("fc2", nn.Linear(in_features=800, out_features=500)),
            ("relu3", nn.ReLU()),
            ("fc3", nn.Linear(in_features=500, out_features=CLASSES_AMOUNT)),
            ("softmax", nn.Softmax(dim=1))
        ]))

        # Initialize a dictionary for the training history
        self._training_history = {
            "train_loss": [],
            "train_acc": []
        }

        self._idx_to_class = {}  # Initialize classes mappings. This will be set by load or by save_classes_info.
        self._classes: list[str] = []  # Initialize classes names. This will be set by load or by save_classes_info.

    def train(self, train_loader):
        """
        The training loop for the model.
        """
        # Put the model into training mode
        self._model.train()

        train_steps = len(train_loader.dataset) // BATCH_SIZE  # Calculate the amount of iterations in each epoch.

        optimizer = optim.Adam(self._model.parameters(), lr=INITIAL_LEARNING_RATE)  # Initialize the optimizer
        loss_function = nn.CrossEntropyLoss()  # Initialize the cost function

        start_time = time()  # Store time when starting training
        for epoch in range(1, EPOCHS + 1):
            # Initialize the total training loss and the number of correct predictions in the training
            total_training_loss = 0
            train_correct = 0

            # Wrap the train loader with a progress bar, for visual progress indication
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{EPOCHS}") as train_loader_progress:
                # Loop over the entire training set, in batches
                for step, (inputs, labels) in enumerate(train_loader_progress):
                    # Put the inputs and labels on the correct device (gpu/cpu):
                    inputs = inputs[0].to(self.device)
                    labels: torch.Tensor = labels[1].to(self.device)

                    # Preform a forward pass:
                    predictions = self._model(inputs)

                    loss = loss_function(predictions, labels)  # Calculate the loss

                    # Backpropagation:
                    optimizer.zero_grad()  # Reset the gradients
                    loss.backward()  # Calculate new gradients
                    optimizer.step()  # Update the weights

                    # add the loss to the total training loss so far
                    total_training_loss += loss
                    # calculate the number of correct predictions
                    train_correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
                    # Display loss on the progress bar
                    train_loader_progress.set_postfix(loss=loss.item())

            avg_train_loss = total_training_loss / train_steps  # Calculate the average training loss
            train_correct /= len(train_loader.dataset)  # Calculate the training accuracy

            # Update the history:
            self._training_history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            self._training_history["train_acc"].append(train_correct)
            end_time = time()  # Store time finished training
            print(f"Total time taken to train the model: {end_time - start_time:.2f}s")

        # Save the data about the class labels from the dataloader.
        self._save_classes_info(train_loader)

    def test(self, test_loader):
        """
        Test the model on a test dataset.
        """
        # Put the model into prediction mode
        self._model.eval()
        # Initialize a list to store predictions:
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing the model", unit="batch"):
                inputs = batch[0].to(self.device)

                # Preform a forward pass (predictions)
                prediction = self._model(inputs)
                # Append the predictions to the predictions array.
                predictions.extend(prediction.argmax(axis=1).cpu().numpy())

        # Generate Reports:
        report = classification_report(
            np.array(test_loader.dataset.targets), np.array(predictions), target_names=self._classes
        )
        print(report)

        print(f"Accuracy: {accuracy_score(np.array(test_loader.dataset.targets), np.array(predictions))}")

        # Generate a confusion matrix
        cm = confusion_matrix(np.array(test_loader.dataset.targets), np.array(predictions), normalize="all")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._classes)
        disp.plot(include_values=False)
        plt.show()

    def predict(self, inputs: torch.Tensor) -> list[str]:
        """
        Predict the class of a given input.
        Input shape has to be (n, 1, 32, 32) for n inputs.
        :return: The predicted class
        """
        # Put the model into prediction mode
        self._model.eval()
        with torch.no_grad():
            # Generate predictions
            predictions = self._model(inputs)
            # Get the prediction with the highest activation
            predictions = predictions.argmax(axis=1).cpu().numpy().tolist()

        # Map the returned indexes to characters
        return [self._idx_to_class[prediction] for prediction in predictions]

    @staticmethod
    def load(path: str) -> "Model":
        """
        Load a previously trained model
        :param path: Path to the trained model
        """
        m = Model()  # Initialize a new model
        data = torch.load(path, map_location=Model.device)  # Load model data
        m._model.load_state_dict(data["state_dict"])  # Load model parameters
        m._model.to(Model.device)  # Put model on the correct device (cpu/gpu)
        # Load class mappings and names
        m._idx_to_class = data["idx_to_class"]
        m._classes = list(m._idx_to_class.values())

        return m

    def save(self, path: str):
        """
        Save a trained model
        :param path: Path to the file (usually ending in .pt) the model will be saved in
        """
        torch.save({
            "state_dict": self._model.state_dict(),  # Save model parameters
            "idx_to_class": self._idx_to_class  # Save class mappings and names
        }, path)
        print(f"Saved model successfully in {path}")

    def _save_classes_info(self, loader):
        # The character '<' is mapped as "arrow", because '<' can't be a folder name.
        # Map it back to the original '<' for usage.
        # Index to class mappings:
        self._idx_to_class = {
            value: key if key != 'arrow' else '<' for key, value in loader.dataset.class_to_idx.items()
        }
        self._classes = list(self._idx_to_class.values())  # Class names

    def __repr__(self):
        """
        When printing the instance, show the string representation of the
        model the instance is wrapping.
        """
        return self._model.__repr__()
