from PIL import Image
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw
from depict import getCDKDepiction


def get_display(img_path: str, smiles: str, size=400):
    """This fucntion takes a image path and
    corresponding SMILES string of a molecule
    and returns a display
    Args (str): image path, SMILES string
    Returns (IPython): display
    """

    segmented_image = Image.open(img_path)
    # Get the original dimensions of the image
    width, height = segmented_image.size

    # Determine the aspect ratio of the image
    aspect_ratio = width / height

    # Calculate the new height based on the desired width of 400 pixels and the aspect ratio
    new_height = int(400 / aspect_ratio)

    # Resize the image with the calculated dimensions
    resized_image = segmented_image.resize((400, new_height))

    # Load the molecule and render the getCDKDepiction image
    depicted_image = getCDKDepiction(smiles, molSize=(size, size))

    # Get the dimensions of the two images
    width1, height1 = resized_image.size
    width2, height2 = depicted_image.size

    # Determine the maximum height of the two images
    max_height = max(height1, height2)

    # Create a new image with the dimensions of the two images side by side
    new_image = Image.new("RGB", (width1 + width2, max_height), color=(255, 255, 255))

    # Calculate the y-coordinate for the center of the composite image
    center_y = int((max_height - height1) / 2)

    # Paste the two images side by side
    new_image.paste(resized_image, (0, center_y))
    new_image.paste(depicted_image, (width1, 0))

    return display(new_image)
