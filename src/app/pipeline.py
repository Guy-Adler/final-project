from src.config import IMG_SIZE, LINES_AMOUNT, CHARACTERS_AMOUNT
# region types
# This includes types to make the IDE code completion better.
from typing import Callable
from .utils import Array


# endregion


class Pipeline:
    """
    The main pipeline, from getting the input data to writing the output file.
    """

    def __init__(self,
                 set_heading: Callable = lambda *x: None,
                 set_progress: Callable = lambda *x: None,
                 set_filename: Callable = lambda *x: None,
                 set_task: Callable = lambda *x: None,
                 log: Callable = lambda *x: None,
                 view_contours: bool = False,
                 view_debug: bool = False
                 ):
        """
        Initialize the instance variables for the pipeline methods.

        All of the imports are done here, **after startup**, to prevent long startup times.
        :param set_heading: The method to set the heading in the GUI
        :param set_progress: The method to set the progress in the GUI
        :param set_filename: The method to set the filename in the GUI
        :param set_task: The method to set the task in the GUI
        :param log: The method to add a log to the GUI
        :param view_contours: Whether to show the contours of the detections
        :param view_debug: Whether to show the segmentation process
        """
        # Keep all the GUI function
        self._set_heading = set_heading
        self._set_progress = set_progress
        self._set_filename = set_filename
        self._set_task = set_task
        self._log = log

        # Initialize the GUI for the loading process
        self._set_heading(f'Getting Ready...')
        self._set_progress(0)
        total_tasks = 15

        # region Import required modules (importing takes time, so better to do it after the startup)
        self._set_filename("Loading image processing...")
        import cv2
        # Load cv2 for _segment
        self.cv2 = cv2
        self._set_progress(1 / total_tasks)

        from .segmentation import LineSegmentor, CharacterSegmentor
        # Initialize segmentors for _segment
        self._line_segmentor = LineSegmentor(draw_contours=view_contours, draw_debug=view_debug)
        self._char_segmentor = CharacterSegmentor(draw_contours=view_contours, draw_debug=view_debug)
        self._set_progress(3 / total_tasks)

        from .utils import get_images_by_path, count_images
        # Load file system functions for pipeline
        self.get_images_by_path = get_images_by_path
        self.count_images = count_images
        self._set_progress(5 / total_tasks)

        from numpy import array
        # Load numpy for _segment
        self.np_array = array
        self._set_progress(6 / total_tasks)

        self._set_filename("Loading model...")
        from torch import device, cuda, tensor, float as torch_float
        # Load required pytorch functions for _predict
        self.torch_device = device
        self.torch_cuda = cuda
        self.torch_tensor = tensor
        self.torch_float = torch_float
        self._set_progress(10 / total_tasks)

        from ..model.model import Model
        from src.config import MODEL_PATH
        # Load and initialize model for _predict
        self._model = Model.load(MODEL_PATH)

        self._set_progress(11 / total_tasks)

        self._set_filename("Loading parser...")
        from .parsing import Parser
        # Load parser for _parse
        self.Parser = Parser
        self._set_progress(12 / total_tasks)

        self._set_filename("Loading file writer...")
        from tempfile import TemporaryFile
        # Load temporary file manager for _generate_csv
        self.TemporaryFile = TemporaryFile
        self._set_progress(13 / total_tasks)

        from csv import reader, writer
        # Load csv reader and writer for _generate_csv, _add_csv_row, _save_csv
        self.csv_reader = reader
        self.csv_writer = writer
        self._set_progress(15 / total_tasks)
        # endregion

    def _segment(self, image: Array, path: str) -> Array | None:
        """
        Segment the image and get images of all 88 characters
        :param image: The image to segment
        :param path: Path to the image
        :return: An array of all character images
        """
        self._set_task(f'Segmenting Characters...')  # Update GUI

        lines = self._line_segmentor.segment(image)  # Get the MRZ lines
        ready_characters = []  # Will keep the characters ready for prediction
        if lines is None or len(lines) != LINES_AMOUNT:
            # Not all lines were found. Ignore the passport.
            if lines is None:
                prefix = "No"
            else:
                prefix = f'{len(lines)} != {LINES_AMOUNT}'
            self._log(f'{prefix} lines were found in {path}', 'error')
            return None

        for j, line in enumerate(lines):
            chars = self._char_segmentor.segment(line)  # Get the characters
            if len(chars) != CHARACTERS_AMOUNT:
                # Not all characters were found. Ignore the passport.
                self._log(f'Not all characters were found in {path}', 'error')
                return None
            for c in chars:
                # Get each character ready for prediction
                char = self.cv2.cvtColor(c, self.cv2.COLOR_BGR2GRAY)  # Convert the character to grayscale
                char = self.cv2.resize(char, (IMG_SIZE, IMG_SIZE))  # Resize the character to 32x32
                # Convert the image to a binary image
                char = self.cv2.threshold(char, 255 // 2, 255, self.cv2.THRESH_BINARY | self.cv2.THRESH_OTSU)[1]

                char = [char]  # Add another dimension (the model expects the image to be 1x32x32)
                ready_characters.append(char)  # Append the character to the list of characters ready for the model

        return self.np_array(ready_characters)

    def _predict(self, chars: Array) -> str:
        """
        Predict the character in the image(s).

        :param chars: An array of characters. Shape should be 88x1x32x32, but in theory all batch sizes can work.
        :return: The MRZ string.
        """
        self._set_task(f'Predicting Characters...')  # Update GUI
        # Get the device to use for the model. If the GPU is available, use it; it is faster.
        device = self.torch_device("cuda" if self.torch_cuda.is_available() else "cpu")
        # Convert the characters into a tensor (base unit of pytorch), and run the model
        predictions = self._model.predict(self.torch_tensor(chars, dtype=self.torch_float).to(device))
        # Concatenate all characters to the MRZ string.
        return ''.join(predictions)

    def _parse(self, mrz_text: str, path: str):
        """
        Build and run the MRZ parser, to get the passport data
        :param mrz_text: The MRZ string to parse
        :param path: Path to the image
        :return: The parser object
        """
        self._set_task(f'Parsing MRZ...')  # Update GUI
        parser = self.Parser(mrz_text)  # Run parser

        # Log parser warnings
        meta = parser.get_meta()
        if meta['should_check_mrz']:
            self._log(f"The passport might not have been parsed correctly for {path}. Please Check it.", 'warning')
        else:
            if meta['should_check_name']:
                self._log(f"The name might not have been parsed correctly for {path}. Please Check it.", 'warning')
            if meta['should_check_passport_number']:
                self._log(f"The passport number might not have been parsed correctly for {path}. Please Check it.",
                          'warning')
            if meta['should_check_date_of_birth']:
                self._log(f"The date of birth might not have been parsed correctly for {path}. Please Check it.",
                          'warning')
            if meta['should_check_date_of_expiry']:
                self._log(f"The date of expiry might not have been parsed correctly for {path}. Please Check it.",
                          'warning')
            if meta['should_check_personal_number']:
                self._log(f"The personal number might not have been parsed correctly for {path}. Please Check it.",
                          'warning')

        return parser

    def _generate_csv(self):
        """
        Generate a temporary CSV file, and add headers to it.
        :return: A pointer to the temporary file.
        """
        # The headers (first line) of the CSV file:
        headers = [
            'File',
            'Issuing Country',
            'Last Name',
            'First Name',
            'Passport Number',
            'Nationality',
            'Date of Birth',
            'Sex',
            'Date of Expiry',
            'Personal Number',
        ]
        file = self.TemporaryFile(mode="w+", newline='')  # Create the temporary file, with both read and write access.
        writer = self.csv_writer(file)  # Create the csv writer
        writer.writerow(headers)  # Write the headers
        return file

    def _add_csv_row(self, file, image_path: str, parser):
        """
        Add a row of passport details to the csv file.
        :param file: A pointer to the temporary file
        :param image_path: The path to the passport image
        :param parser: The parsed MRZ object
        """

        def to_date(date) -> str:
            """
            Build a date format from the parser date. Using '.' as separators to prevent Excel from recognizing
            it as a date, and making a mess. If a part of the date is unknown, it will be shown as '00'.
            :param date: A parser date object
            :return: A date string
            """
            return f'{date.get("day")}.{date.get("month")}.{date.get("year")}'

        writer = self.csv_writer(file)  # Create the writer
        data = parser.get_data()  # Get the MRZ data from the parser object

        # Write the passport data as a row to the csv
        writer.writerow([
            image_path,
            data['issuer_iso'],
            data['name']['last'],
            data['name']['first'],
            data['passport_number'],
            data['nationality_iso'],
            to_date(data['date_of_birth']),
            data['sex'],
            to_date(data['date_of_expiry']),
            data["personal_number"]
        ])

    def _save_csv(self, file, output: str):
        """
        Save the temporary csv file to the output location.

        Used https://docs.python.org/3/library/tempfile.html#examples for debugging.
        :param file: A pointer to the temporary csv file.
        :param output: The path to the output file.
        """
        file.seek(0)  # Go back to the top of the file stream

        reader = self.csv_reader(file)  # Create a csv reader for the temporary file
        with open(output, mode="w", newline="") as new_file:  # Create the output file
            writer_obj = self.csv_writer(new_file)  # Create a csv writer for the output file
            # Copy the file to the output location
            for data in reader:
                writer_obj.writerow(data)

    def pipeline(self, path: str, output: str):
        """
        The main pipeline, from getting the input data to
        writing the output file.
        :param path: The path to the folder containing the input images
        :param output: The path to the csv output file
        """
        self._set_progress(0)  # Set the GUI progress bar to 0%
        images = self.get_images_by_path(path)  # Get a generator of all images

        # Get the total amount of images in the input folder
        images_amount = self.count_images(path)
        file = self._generate_csv()  # Get a pointer to the temporary csv file

        for i, (image, image_path) in enumerate(images):
            # Update GUI
            self._set_heading(f'Extracting Data ({i + 1}/{images_amount})')
            self._set_filename(image_path)

            # Start the data pipeline:
            # segmentation -> prediction -> parsing -> appending to output
            chars = self._segment(image, image_path)
            if chars is not None:
                mrz_text = self._predict(chars)
                parsed_mrz = self._parse(mrz_text, image_path)
                self._add_csv_row(file, image_path, parsed_mrz)
                self._log(f"Added passport data from {image_path}")

            self._set_progress((i + 1) / images_amount)  # update GUI

        # Save the temporary output file to the output location
        self._save_csv(file, output)
