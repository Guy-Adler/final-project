import tkinter as tk
import tkinter.ttk as ttk


class Checkbox(ttk.Frame):
    """
    A widget containing a checkbox, with an easier interface.
    """

    def __init__(self, parent: tk.Misc, label_text: str):
        super().__init__(parent)
        self._value = tk.BooleanVar(value=False)  # Create the variable storing the checkbox value. Default off.
        self._input = ttk.Checkbutton(self, text=label_text, variable=self._value)  # Create the checkbox

        self._input.pack()  # Place the checkbox

    def get(self):
        """
        Get the checkbox value
        """
        return self._value.get()

    def reset(self):
        """
        Reset the checkbox
        """
        self._value.set(False)
