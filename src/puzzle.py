import json
from os import listdir
from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
from PIL import Image
from torch import Tensor, stack
from torchvision import transforms


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    with open(path / "metadata.json", "r") as file:
        metadata = json.load(file)
        extrensics = Tensor(metadata["extrinsics"])
        intrinsics = Tensor(metadata["intrinsics"])

    image_list = []

    for filename in sorted(listdir(path / "images")):
        pil_img = Image.open(path / "images" / filename)
        tensor_img = transforms.ToTensor()(pil_img)
        image_list.append(tensor_img)

    images_tensor = stack(image_list)

    dataset = PuzzleDataset(
        extrinsics=extrensics, intrinsics=intrinsics, images=images_tensor
    )
    return dataset


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    return dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
