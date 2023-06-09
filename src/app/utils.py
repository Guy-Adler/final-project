import cv2
import os
import contextlib
# region types
# This includes types to make the IDE code completion better.
import numpy.typing as npt
from numpy import uint8

Array = npt.NDArray[uint8]
BoundingBox = tuple[int, int, int, int]


# endregion


@contextlib.contextmanager
def show_images():
    """
    A context to open multiple images and wait for user interaction before continuing.

    Used https://www.geeksforgeeks.org/context-manager-using-contextmanager-decorator for help

    Usage example:
        with show_images():
            cv2.imshow("Image1", img1)
            cv2.imshow("Image2", img2)
    """
    # Upon entering the context, do nothing
    yield
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(image: Array, width: int | None = None, height: int | None = None) -> Array:
    """
    Resizes an image while keeping the aspect ratio (prevents stretching).

    Taken from https://stackoverflow.com/a/44659589
    :param image: The image to resize
    :param width: The width to resize to, None if it should be set according to the height.
    :param height: The height to resize to, None if it should be set according to the width.
    :return: The resized image.
    """

    if width is None and height is None:
        # No resizing needed.
        return image

    # Initialize variables for the old size
    old_height: int
    old_width: int
    old_height, old_width = image.shape[:2]

    # Initialize variables for the new size, as well as the resize ratio.
    new_height: int
    new_width: int
    resize_ratio: float

    if width is None:
        # Resize based on the height
        new_height = height

        # Calculate the resize factor, to keep the aspect ratio:
        resize_ratio = new_height / float(old_height)

        # Calculate the new width:
        new_width = int(old_width * resize_ratio)

    else:
        # Resize based on the width
        new_width = width

        # Calculate the resize factor, to keep the aspect ratio:
        resize_ratio = new_width / float(old_width)

        # Calculate the new width:
        new_height = int(old_height * resize_ratio)

    # According to the OpenCV docs, when zooming an image (resize factor > 1)
    # one should use the INTER_CUBIC or INTER_LINEAR interpolation (using INTER_LINEAR is faster),
    # whereas when shrinking an image (resize factor < 1) one should use the INTER_AREA interpolation.
    interpolation = cv2.INTER_LINEAR if resize_ratio > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")  # All extensions considered to be images


def get_images_by_path(path: str):
    """
    Create a generator to load each image in a given folder, and its subdirectories.
    Creating a generator means not all images are loaded to memory at once, giving better performance
    and allowing to load a folder with a lot of images even on computers without a lot of memory.

    The skeleton is taken from https://stackoverflow.com/a/59925514

    :param path: The path to the root folder.
    """
    for root, dirs, files in os.walk(path):  # Traverse the directories recursively
        for filename in files:
            # Sometimes the file extension is not saved in all lowercase, so it needs to be converted.
            if os.path.splitext(os.path.join(root, filename))[1].lower() in IMAGE_EXTENSIONS:
                # The file is an image. Load it.
                image: Array = cv2.imread(os.path.join(root, filename))
                # *yield* the image and filename. This keeps the generator running, but returns the data
                # to the user.
                yield image, filename


def count_images(path: str) -> int:
    """
    Count the amount matching files in a given folder
    :param path: The path to the folder.
    """
    count = 0  # Initialize the file count

    for root, dirs, files in os.walk(path):  # Traverse the directories recursively
        for filename in files:
            # Sometimes the file extension is not saved in all lowercase, so it needs to be converted.
            if os.path.splitext(os.path.join(root, filename))[1].lower() in IMAGE_EXTENSIONS:
                # The file is an image. Increase the counter.
                count += 1
    return count


def get_image_by_path(path: str) -> list[tuple[Array, str]]:
    """
    Load an image. Returns in the same format as get_images_by_path, for ease of use.
    :param path: The path for the image.
    """
    return [(cv2.imread(path), os.path.basename(path))]
