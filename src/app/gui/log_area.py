import tkinter as tk
import tkinter.scrolledtext as tkst

# region types
# This includes types to make the IDE code completion better.
DumpedData = list[tuple[str, str, str]]


# endregion


class LogArea(tkst.ScrolledText):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent, state=tk.DISABLED)
        self.tag_config('error', foreground="red")
        self.tag_config('warning', foreground="orange")

    def get(self) -> DumpedData:
        return self.dump('1.0', tk.END, tag=True, text=True)

    def load(self, dumped: DumpedData):
        # https://stackoverflow.com/a/74198450
        tags: list[str] = []
        self.configure(state=tk.NORMAL)  # Enable log area editing
        for (key, value, index) in dumped:
            if key == "tagon":
                # A tag is activated. Append it to the list of active tags
                tags.append(value)
            elif key == "tagoff":
                # A tag is deactivated. Remove it from the list of active tags
                tags.remove(value)
            elif key == "text":
                self.insert(tk.INSERT, value, tags)  # Insert the text
        self.configure(state=tk.DISABLED)  # Disable log area editing
        self.see(tk.END)  # Scroll to the end of the log area

    def pack_forget(self):
        """
        Unload the element from view, and reset its data.
        """
        self.configure(state=tk.NORMAL)
        self.delete('1.0', tk.END)  # https://stackoverflow.com/a/27967664
        self.configure(state=tk.DISABLED)
        super().pack_forget()

    def add_log(self, text: str, tag: str | None = None):
        """
        Add a new log line(s) to the log screen
        :param text: The line(s) to add
        :param tag: The tag to add to the added text
        """
        self.configure(state=tk.NORMAL)  # Enable log area editing

        # Make sure the text ends with a newline:
        if not text.endswith('\n'):
            text += '\n'

        self.insert(tk.INSERT, text, tag)  # Insert the text
        self.configure(state=tk.DISABLED)  # Disable log area editing
        self.see(tk.END)  # Scroll to the end of the log area
