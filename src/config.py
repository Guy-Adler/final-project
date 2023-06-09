from os import path

IMG_SIZE = 32  # The width/height of the generated images
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"  # The characters to generate/predict
FONT_SIZE = 32  # The font size for the generated images
AMOUNT = 10_000  # The amount of images to generate
TRAIN_SPLIT = 0.8  # The percentage of generated images to put in the training dataset (the rest goes to test)
BATCH_SIZE = 256  # The batch size (amount of images the model trains on at once)
INITIAL_LEARNING_RATE = 1e-3  # The initial learning rate for the optimizer
EPOCHS = 20  # Amount of repeats of training on the entire dataset
CLASSES_AMOUNT = len(CHARACTERS)  # Amount of classes to classify between
BASE_PATH = path.join('resources', 'data', '')  # Base path for the data
MODEL_PATH = path.join('resources', 'model.pt')  # Path to save/load the model to
LINES_AMOUNT = 2  # Amount of lines in an MRZ
CHARACTERS_AMOUNT = 44  # Amount of characters in an MRZ line
