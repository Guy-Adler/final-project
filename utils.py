import cv2


def show_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
