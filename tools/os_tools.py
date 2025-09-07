import argparse

def image_size_arg(string: str):
    try:
        # Split by comma and convert to integers
        values = tuple(map(int, string.split(',')))
        if len(values) != 2:
            raise argparse.ArgumentTypeError("Must be exactly two values separated by a comma")
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("Must be integers separated by a comma")