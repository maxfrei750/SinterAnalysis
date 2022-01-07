import fire

from external.paddle.data_preparation.utilities import split


def split_dataset(root, validation_percentage: float = 0.25):
    split(root, validation_percentage)


if __name__ == "__main__":
    fire.Fire(split_dataset)
