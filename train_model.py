from src.model.data_loaders import train_loader, test_loader
from src.model.model import Model
from src.config import MODEL_PATH

if __name__ == '__main__':
    model = Model()  # Initialize an untrained model
    model.train(train_loader)  # Train the model on the training data
    model.save(MODEL_PATH)  # Save the model
    model.test(test_loader)  # Test the model
