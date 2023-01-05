import contextlib
import cv2
import os
# Types
import numpy.typing as npt
from numpy import uint8

Array = npt.NDArray[uint8]
BoundingBox = tuple[int, int, int, int]


@contextlib.contextmanager
def show_images():
    """
    A context to open multiple images and wait for user interaction before continuing.


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
    if width is None and height is None:
        # No resizing needed.
        return image

    # Initialize variables for the old size
    old_height: int
    old_width: int
    old_height, old_width = image.shape[:2]

    # Initialize variables for the new size, as well as the resize ratio
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


def get_images_by_path(path):
    # return the set of files that are valid
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                # construct the path to the image and yield it
                yield cv2.imread(os.path.join(rootDir, filename))


def get_image_by_path(path):
    return cv2.imread(path)
