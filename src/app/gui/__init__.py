from ctypes import windll
import tkinter as tk
import threading
from ..pipeline import Pipeline
from .main_page import MainPage
from .progress_page import ProgressPage
from .finished_page import FinishedPage


class Program:
    """
    The main program class.
    Contains the GUI screens and the pipeline initialization.
    """

    def __init__(self):
        # Fix blurry screen on some devices (https://stackoverflow.com/a/43046744)
        windll.shcore.SetProcessDpiAwareness(1)
        # Initialize main window
        self._root = tk.Tk()
        self._root.title('Passports Data Extractor')  # Set window title
        self._root.geometry('800x500')  # Set window size

        # Initialize variables for the program:
        self._folder_path = ''
        self._output_file = ''
        self._view_contours = False
        self._view_debug = False

        # Initialize pages
        self._main_page = MainPage(self._root, self._on_start_press)
        self._progress_page = ProgressPage(self._root)
        self._finished_page = FinishedPage(self._root, self._return_to_main_page, self._root.destroy)

        # Display the main page to the screen
        self._main_page.pack()

        # Start the main GUI loop
        self._root.mainloop()

    def _on_start_press(self):
        """
        Runs when the start button on the main screen is clicked.
        Saves the data from the screen, and starts the pipeline.
        """
        values = self._main_page.get()
        self._main_page.pack_forget()  # Remove the main page from the screen

        # Save the values of the main screen to the instance
        self._folder_path = values['folder_path']
        self._output_file = values['output_file']
        self._view_contours = values['view_contours']
        self._view_debug = values['view_debug']

        # Display the progress page
        self._progress_page.pack(fill=tk.BOTH, expand=True)

        # Start a new thread for the pipeline, so it won't freeze the event loop of the main program screen.
        # Start it in daemon mode, so it will stop executing when the main program is stopped (the window is closed).
        threading.Thread(target=self._start_pipeline, daemon=True).start()

    def _return_to_main_page(self):
        """
        Returns the GUI to the main page
        """
        self._finished_page.pack_forget()  # remove the finished page
        self._main_page.pack()  # display the main page

    def _start_pipeline(self):
        """
        Start the main pipeline process, with GUI updates.
        """
        # Create a pipeline instance
        p = Pipeline(
            set_heading=self._progress_page.set_heading,
            set_progress=self._progress_page.set_progress,
            set_filename=self._progress_page.set_filename,
            set_task=self._progress_page.set_task,
            log=self._progress_page.add_log,
            view_contours=self._view_contours,
            view_debug=self._view_debug,
        )

        # Execute the pipeline
        p.pipeline(self._folder_path, self._output_file)

        # After the pipeline finished executing, copy to log from the progress page to the finished page:
        log_text = self._progress_page.get_log()  # Get the log text
        self._finished_page.set_log(log_text)  # Paste the log text on the finished page
        self._progress_page.pack_forget()  # Remove the progress page
        self._finished_page.pack(fill=tk.BOTH, expand=True)  # Display the finished page
