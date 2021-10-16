import torch


def generateRandomVectors(n, min, max, dtype):
    """Generate a set of random vectors of shape [n, len(min)] with values from [min, max].
    Arguments:
        n          - the number of vectors generated; int
        min        - minimum value for each element of the vector; iterable
        max        - maximum value for each element of the vector; iterable
        dtype      - data type of the returned vectors; torch.dtype
    Raises:
        ValueError - len(min) is not equal to len(max) or min > max for at least one element
    """
    if len(min) != len(max):
        raise ValueError(f"Different lengths of min and max. Expected len(min) == len(max); got len(min)={len(min)}, len(max)={len(max)}.")

    min = torch.tensor(min, dtype=dtype)
    max = torch.tensor(max, dtype=dtype)

    if not (min < max).all():
        raise ValueError(f"min < max not satisfied. Expected all True, got {(max > min)}.")

    range = max - min
    shape = torch.Size([n]) + range.shape
    vecs = torch.rand(shape, dtype=dtype) * range + min
    return vecs


def generateUnitVectors(shape, dtype):
    """Generate a set of random unit vectors of shape.
    Arguments:
        shape - the shape of the form (n, d) where n is number of vectors in the set and d is their dimension
        dtype - data type of the returned vectors; torch.dtype
    """
    vecs = 2 * torch.rand(shape, dtype=dtype) - 1
    vecs = torch.einsum('ni, n -> ni', vecs, (1 / vecs.norm(dim=1)))
    vecs.norm(dim=1).sum()
    return vecs
