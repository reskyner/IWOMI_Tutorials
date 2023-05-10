import os
from pdf2image import convert_from_path


def convert_pdf_to_images(pdf_path: str):
    """
    This function takes the path of a pdf file, converts the pages to 300 DPI
    images, saves the resulting PNG files in the directory of the input pdf
    file and returns a list of paths
    Args:
        pdf_path (str): path of pdf file
    Returns:
        (list): list of path to the images
    """
    directory = os.path.join(f"{os.getcwd()}", pdf_path + "_output")
    if not os.path.exists(directory):
        os.makedirs(directory)

    page_images = convert_from_path(pdf_path, 300)
    im_paths = []
    num = 0
    for image in page_images:
        im_path = "{}_{}.png".format(pdf_path[:-3], num)
        full_im_path = os.path.join(directory, im_path)
        image.save(full_im_path, format="PNG")
        im_paths.append(full_im_path)
        num += 1

    im_paths.sort()
    return im_paths
