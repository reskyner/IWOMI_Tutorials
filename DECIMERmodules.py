import os
import cv2
from PIL import Image
from decimer_segmentation import (
    segment_chemical_structures,
    segment_chemical_structures_from_file,
)
from DECIMER import predict_SMILES


def convert_image(path: str):
    """Takes an image filepath of GIF image and returns Hi Res PNG image.
    Args:
        input_path (str): path of an image.
    Returns:
        segment_paths (list): a list of paths of segmented images.
    """
    img = Image.open(path).convert("RGBA")
    new_size = int((float(img.width) * 2)), int((float(img.height) * 2))
    resized_image = img.resize(new_size, resample=Image.LANCZOS)
    background_size = int((float(resized_image.width) * 2)), int(
        (float(resized_image.height) * 2)
    )
    new_im = Image.new(resized_image.mode, background_size, "white")
    paste_pos = (
        int((new_im.size[0] - resized_image.size[0]) / 2),
        int((new_im.size[1] - resized_image.size[1]) / 2),
    )
    new_im.paste(resized_image, paste_pos)
    new_im.save(path.replace("gif", "png"), optimize=True, quality=100)
    return path.replace("gif", "png")


def get_segments(path: str):
    """Takes an image filepath and returns a set of paths and image name of segmented images.
    Args:
        input_path (str): path of an file.
    Returns:
        image_name (str): image file name.
        segments (list): a set of segmented images.
    """
    file_name = os.path.split(path)[1]
    if file_name[-3:].lower() == "pdf":
        segments = segment_chemical_structures_from_file(
            path, expand=True, poppler_path=None
        )
        return file_name, segments
    else:
        page = cv2.imread(path)
        segments = segment_chemical_structures(page, expand=True)
        return file_name, segments


def getPredictedSegments(path: str):
    """Takes an image filepath and returns predicted SMILES for segmented images.
    Args:
        input_path (str): path of an image.
    Returns:
        predictions (list): a list of SMILES of the segmented images.
    """
    smiles_predicted = []
    segment_paths = []
    image_name, segments = get_segments(path)

    segment_directory = os.path.join(
        os.path.split(path)[0], os.path.split(path)[1][:-4] + "segments"
    )
    if not os.path.exists(segment_directory):
        os.makedirs(segment_directory)
    if len(segments) == 0:
        # smiles = Predictor_exported.predict_SMILES(path)

        return "No segments"
    else:
        for segment_index in range(len(segments)):
            segmentname = f"{image_name[:-4]}_{segment_index}.png"
            segment_path = os.path.join(segment_directory, segmentname)
            segment_paths.append(segment_path)
            cv2.imwrite(segment_path, segments[segment_index])
            smiles = predict_SMILES(segment_path)
            smiles_predicted.append(smiles)
        return segment_paths, smiles_predicted
