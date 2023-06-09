import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkf
from .log_area import LogArea


class ProgressPage(ttk.Frame):
    """
    A page showing the progress of the main pipeline.
    """

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)

        # Initialize and place the heading
        self._heading_text = tk.StringVar(value="Getting Ready...")  # This stores the heading text. It will be changed.
        self._heading = ttk.Label(self, textvariable=self._heading_text, font=tkf.Font(size=20, weight="bold"))
        self._heading.pack()

        # Initialize and place the progress bar
        self._progress = ttk.Progressbar(self, mode='determinate')
        self._progress.pack(fill=tk.X)

        # Initialize and place the current file label
        self._file_text = tk.StringVar()  # This stores the text of the file label. It will be changed.
        self._file = ttk.Label(self, textvariable=self._file_text)
        self._file.pack()

        # Initialize and place the current task label
        self._task_text = tk.StringVar()  # This stores the text of the current task. It will be changed.
        self._task = ttk.Label(self, textvariable=self._task_text)
        self._task.pack()

        # Initialize and place the log area, where logs are printed.
        self._log_area = LogArea(self)

        self._log_area.pack(fill=tk.BOTH, expand=True)

    def pack_forget(self):
        """
        Unload the page from view, and reset its data.
        """
        self.set_heading("Getting Ready...")
        self.set_progress(0)
        self.set_filename("")
        self._log_area.pack_forget()
        super().pack_forget()

    def set_heading(self, label: str):
        self._heading_text.set(label)

    def set_progress(self, progress: float):
        self._progress.configure(value=progress * 100)

    def set_filename(self, file: str):
        self._file_text.set(file)

    def set_task(self, task: str):
        self._task_text.set(task)

    def add_log(self, text: str, tag: str | None = None):
        """
        Add a new log line(s) to the log screen
        :param text: The line(s) to add
        :param tag: The tag to add to the added text
        """
        self._log_area.add_log(text, tag)

    def get_log(self):
        return self._log_area.get()
