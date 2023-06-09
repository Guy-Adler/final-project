import tkinter as tk
import tkinter.ttk as ttk
# region types
# This includes types to make the IDE code completion better.
from typing import Callable


# endregion


class InputWithButton(ttk.Frame):
    """
    A widget containing a label, and entry, and a button. When the button is called, the returned value
    of the executed command is set as the value of the entry.
    """

    def __init__(self, parent: tk.Misc, label_text: str, button_text: str, command: Callable, onchange: Callable):
        super().__init__(parent)
        self._command = command  # The command to be executed on button press
        self._onchange = onchange  # A callback to be called on input change

        self._input_var = tk.StringVar()  # Create the variable storing the data
        self._input_var.trace_add('write', lambda *args: onchange())  # Register the onchange function

        # Create the widgets
        self._label = ttk.Label(self, text=label_text)  # The label of the input
        self._input_entry = ttk.Entry(self, textvariable=self._input_var)  # The entry
        self._button = ttk.Button(self, text=button_text, command=self._execute_command)  # The button

        # Place the widgets side by side
        self._label.pack(side=tk.LEFT, padx=2, pady=2)
        self._input_entry.pack(side=tk.LEFT, padx=2, pady=2)
        self._button.pack(padx=2, pady=2)

    def _execute_command(self):
        """
        Run the command provided to the instance, and set the input value to the returned value of the command
        """
        result = self._command()  # Call the command
        if result is not None:  # Make sure the operation was not canceled
            # Set the input value to the result
            self._input_var.set(result)

    def get(self):
        """
        Get the input value
        """
        return self._input_var.get()

    def reset(self):
        """
        Reset the input
        """
        self._input_var.set('')
