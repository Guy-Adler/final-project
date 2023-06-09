from .segmentor import Segmentor, BoundingBox, sort_contours
from ..utils import resize_image, Array
from src.config import LINES_AMOUNT


class LineSegmentor(Segmentor):
    """
    A segmentor for getting the MRZ lines from a picture of a passport
    """

    def __init__(self, draw_contours: bool = False, draw_debug: bool = False):
        super().__init__(draw_contours, draw_debug)
        self._found_rois: int = 0  # The amount of ROIs already selected

    def segment(self, image) -> tuple[Array, Array] | None:
        """
        The segmentation for lines.
        :param image: The image to segment
        :return: a list of all lines
        """
        self._found_rois = 0  # reset to 0 so the same segmentor could be used multiple times.

        # Shrink the image to width of maximum 300px
        resize_width = 300
        # Keep the resize ratio to scale the ROIs back up
        resize_ratio = 1.0 if image.shape[1] <= resize_width else resize_width / image.shape[1]
        resized_image = image if image.shape[1] <= resize_width else resize_image(image, width=resize_width)

        # Call the base segmentor
        rois = super().segment(resized_image)

        if len(rois) < LINES_AMOUNT:  # Check if all ROIs (lines) were found.
            return None

        bottom_line, top_line = rois  # The ROIs are sorted from bottom to top

        # Get the bounding *lines* of the ROIs for the original image size:
        bottom_x_left, bottom_x_right, bottom_y_top, bottom_y_bottom = self.resize_roi(bottom_line[1], resize_ratio)
        top_x_left, top_x_right, top_y_top, top_y_bottom = self.resize_roi(top_line[1], resize_ratio)

        # Get the ROIs in the original image size
        resized_bottom_line = image[bottom_y_top:bottom_y_bottom, bottom_x_left:bottom_x_right].copy()
        resized_top_line = image[top_y_top:top_y_bottom, top_x_left:top_x_right].copy()

        return resized_top_line, resized_bottom_line

    def _post_transformations(self):
        """
        The special transformations for the line segmentor.

        During the thresholding, it's possible that border pixels were connected to the main regions of interest.
        To remove them, set 3% of the borders to zero.
        """
        height, width, _ = self._image.shape  # Get the height and width of the image
        remove_percentage = 0.03

        # Calculate the amount of pixels to remove from the border of the image
        remove_amount = int(height * remove_percentage)

        self._thresh[0:remove_amount, :] = 0  # Remove top border
        self._thresh[height - remove_amount:] = 0  # Remove bottom border
        self._thresh[:, 0:remove_amount] = 0  # Remove left border
        self._thresh[:, width - remove_amount:] = 0  # Remove right border

    def _sort_contours(self):
        """
        Sort the contours from the bottom to the top, since the MRZ lines are at the bottom of the passport.
        """
        self._contours = sort_contours(self._contours, "btt")

    def _is_roi(self, bounding_box: BoundingBox):
        """
        Determine whether the bounding rect belongs to a ROI.

        Conditions are:

        - Max. 2 ROIs per image
        - aspect ratio > 5
        - width is at least 70% of original image

        :param bounding_box: The bounding box of the ROI
        :return: True if the contour should be a ROI, False otherwise.
        """
        if self._found_rois >= LINES_AMOUNT:
            # Limit to only 2 ROIs per image
            return False

        # Calculate the aspect ratio and coverage ratio of the ROI
        *_, width, height = bounding_box  # Get the width and height of the rect
        image_width = self._image.shape[1]  # Get the width of the original image
        aspect_ratio = width / float(height)  # Calculate the aspect ratio (width / height)
        coverage_ratio = width / float(image_width)  # Calculate the ratio between the width of the rect and the image

        minimum_aspect_ratio = 5
        minimum_coverage_ratio = 0.7

        if aspect_ratio > minimum_aspect_ratio and coverage_ratio > minimum_coverage_ratio:
            # It is a ROI
            self._found_rois += 1
            return True

        return False
