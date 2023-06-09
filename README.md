# Passports Parser

## Installation requirements:

- Python > 3.10.9
- Required packages (can be installed with `pip install -r requirements.txt`)

## Creating and training a model:

- To create learning and testing data, run `generate_data.py`.
- To train, test and save a model based on the data, run `train_model.py`.


## Using the software

- Run `main.py` to launch the application.
- Select an input folder with passport images
- Select an output file for the passport data
- Optionally toggle the show detection contours/show detection process checkboxes
- Press start

### Possible errors and warnings

- Sometimes, the program cannot detect a passport. This results in a **red** error in the logs.

- Sometimes, the program makes a mistake in the detection process.
While it may not know how to fix it, it does know a mistake has been made.
In this case, it will notify the user that manual checks are required using a **orange** warning in the logs.