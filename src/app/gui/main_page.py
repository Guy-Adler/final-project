import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkf
import tkinter.filedialog as tkfd
from .input_with_button import InputWithButton
from .checkbox import Checkbox
# region types
# This includes types to make the IDE code completion better.
from typing import Callable


# endregion


def get_output_file() -> str | None:
    """
    Open a dialog to ask for the output file path
    :return: The output file path
    """
    f = tkfd.asksaveasfilename(initialfile='passports.csv', defaultextension=".xlsx",
                               filetypes=[("CSV Documents", "*.csv")], title='Select Output File')
    return f if f != '' else None  # If the operation was cancelled, return None.


def get_input_folder() -> str | None:
    """
    Open a dialog to ask for the input folder path
    :return: The input folder path
    """
    f = tkfd.askdirectory(mustexist=True)
    return f if f != '' else None  # If the operation was cancelled, return None.


class MainPage(ttk.Frame):
    """
    The start screen of the program.
    """

    def __init__(self, parent: tk.Misc, on_start_press: Callable):
        super().__init__(parent)
        # Initialize and place the heading
        self._heading = ttk.Label(self, text='Passports Data Extractor', font=tkf.Font(size=20, weight="bold"))
        self._heading.grid(row=0, columnspan=2)

        # Initialize and place the input folder path input
        self._folder_input = InputWithButton(self, label_text='Folder Path:', button_text='Choose Folder',
                                             command=get_input_folder, onchange=self._set_start_button_state)
        self._folder_input.grid(row=1, columnspan=2)

        # Initialize and place the output file path input
        self._output_input = InputWithButton(self, label_text='Output File:', button_text='Choose File',
                                             command=get_output_file, onchange=self._set_start_button_state)
        self._output_input.grid(row=2, columnspan=2)

        # Initialize and place the contour viewer toggle
        self._view_contours = Checkbox(self, label_text='Show detection contours')
        self._view_contours.grid(row=3, column=0)

        # Initialize and place the segmentation process viewer toggle
        self._view_debug = Checkbox(self, label_text='Show detection process')
        self._view_debug.grid(row=3, column=1)

        # Initialize and place the start button
        self._start_button = ttk.Button(self, text='Start', command=on_start_press, state=tk.DISABLED)
        self._start_button.grid(columnspan=2)

    def pack_forget(self):
        """
        Unload the page from view, and reset its data.
        """
        self._folder_input.reset()
        self._output_input.reset()
        super().pack_forget()

    def get(self):
        """
        Get the page's inputs.
        :return: An object containing the input values of the page
        """
        return {
            'folder_path': self._folder_input.get(),
            'output_file': self._output_input.get(),
            'view_contours': self._view_contours.get(),
            'view_debug': self._view_debug.get()
        }

    def _set_start_button_state(self):
        """
        Make the button disabled if either the folder path or the output path is empty, and normal otherwise.
        Runs on every input change.
        """
        is_empty = self._folder_input.get() == '' or self._output_input.get() == ''  # Some required inputs are empty
        self._start_button.configure(state=tk.DISABLED if is_empty else tk.NORMAL)
