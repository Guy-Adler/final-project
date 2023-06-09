import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkf
from .log_area import LogArea, DumpedData
# region types
# This includes types to make the IDE code completion better.
from typing import Callable


# endregion


class FinishedPage(ttk.Frame):
    """
    An end page, with a quit and return to home options.
    """

    def __init__(self, parent: tk.Misc, return_to_main: Callable, exit: Callable):
        super().__init__(parent)

        # Initialize and place heading
        self._heading = ttk.Label(self, text="Finished!", font=tkf.Font(size=20, weight="bold"))
        self._heading.pack()

        # Initialize a temporary row for the buttons:
        row = ttk.Frame(self)

        # Initialize and place the home button
        self._home_button = ttk.Button(row, text='Return to Main Page', command=return_to_main)
        self._home_button.pack(side=tk.LEFT)

        # Initialize and place the quit button
        self._quit_button = ttk.Button(row, text='Quit', command=exit)
        self._quit_button.pack(side=tk.RIGHT)

        # Place the temporary row
        row.pack()

        # Initialize and place the log area (will be filled later on)
        self._log_area = LogArea(self)
        self._log_area.pack(fill=tk.BOTH, expand=True)

    def set_log(self, log_data: DumpedData):
        """
        Set the value of the log area to `log_text`
        """
        self._log_area.load(log_data)
