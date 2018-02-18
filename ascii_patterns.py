# -*- coding: utf-8 -*-
"""
Find the bug

Finds patters in ascii images

"""

import numpy as np
from itertools import product


def find_pattern(landscape, pattern, rotations=False):
    """
    Count the number of times a pattern (read from a txt file) appers in an ascii image (read from another file)

    Examples
    ---------
    >>> find_pattern('landscape.txt', 'bug.txt')
    3

    Parameters
    ----------
    landscape: str or numpy.array
        Path of the ascii image, or numpy array
    pattern: str or numpy.array
        Path of the pattern or numpy array
    rotations: bool
        If True (defaults to False) the rotated patterns are also matched

    Returns
    -------
    int
        Number of times the pattern appears in the image

    """
    if isinstance(landscape, str):
        landscape = matrix_from_file(landscape)

    if isinstance(pattern, str):
        pattern = matrix_from_file(pattern)

    # If necessary, generate a list with the rotated patterns
    if rotations:
        rotated = generated_rotations(pattern)
        ps = [min(pattern.shape)] * 2
    else:
        rotated = [pattern]
        ps = pattern.shape

    # Look for the pattern
    bugs = 0

    for i, j in product(range(landscape.shape[0] - ps[0] + 1),
                        range(landscape.shape[1] - ps[1] + 1)):

        for r in rotated:
            found_pattern = is_pattern(landscape, r, i, j)

            if found_pattern:
                this_fp = r.shape
                break

        if found_pattern:
            # Increase number of bugs
            bugs += 1

            # Delete the bug from the landscape, to accelerate the search
            landscape[i:i + this_fp[0], j:j + this_fp[1]] = -1

    return bugs


def generated_rotations(pattern):
    """
    Generate a vector with the four rotations of the pattern

    Parameters
    ----------
    pattern: numpy.array

    Returns
    -------
    list

    """

    rotated = [pattern]
    for t in range(3):
        rotated.append(rotated[t].T)

    return rotated


def is_pattern(landscape, pattern, i, j):
    """
    Finds if the pattern start in a given coordanates

    Parameters
    ----------
    landscape: numpy.array
        The data to search in
    pattern: numpy.array
        Pattern to search
    i: int
        Starting x coordinate
    j: int
        Starting y coordinate

    Returns
    -------
    bool

    """

    found_pattern = True

    # Compare element by element until a difference is found
    for x, y in product(range(pattern.shape[0]), range(pattern.shape[1])):
        if landscape[i + x, j + y] != pattern[x, y]:
            found_pattern = False
            break

    return found_pattern


def matrix_from_file(filename):
    """
    Reads a text file and returns a numpy array with a integer matrix representation

    Parameters
    ----------
    filename: str

    Returns
    -------
    numpy.array

    """
    raw_matrix = []

    for line in open(filename):
        char_list = line.rstrip('\n')
        int_list = [ord(c) for c in char_list]
        raw_matrix.append(int_list)

    # All rows need the same number of elements. Insert whitespace (32) at the end
    n_columns = max([len(r) for r in raw_matrix])

    new_matrix = []

    for r in range(len(raw_matrix)):
        rc = len(raw_matrix[r])

        if 0 < rc < n_columns:
            new_matrix.append(raw_matrix[r] + [32] * (n_columns - rc))
        elif rc == 0:
            new_matrix.append([32] * n_columns)
        else:
            new_matrix.append(raw_matrix[r])

    matrix = np.vstack(new_matrix)

    return matrix


def generate_random_landscape(size, pattern, number_of_patterns, rotations=False):
    """
    For testing purposes, generates a random landscape and introduces a pattern a given number of times.

    The function places the pattern randomly, avoiding to stamp on previous patterns. After 2 * number_of_patters
    attempts, it exits with error (this is not a tessellation function, and makes no attempt to be efficient)

    Examples
    ---------
    >>> find_pattern(generate_random_landscape((1000, 1000), 'bug.txt', 100), 'bug.txt')
    100

    Parameters
    ----------
    size: tuple
        Size of the landscape
    pattern: str
        Text file for the pattern
    number_of_patterns: int
        Number of times we want to introduce the pattern into the landscape
    rotations: bool
        If True (defaults to False), the pattern is randomly rotated for each insertion

    Returns
    -------
    np.array

    Raises
    -------
    StopIteration
        If the algorithm is not able to put all the requested patterns

    """

    # Generate random landscape
    landscape = np.random.randint(0, high=1000, size=size)
    pattern_locations = np.zeros(size)
    patterns_introduced = 0
    attempts = 0

    # Read pattern file
    pattern = matrix_from_file(pattern)

    # If necessary, generate a list with the rotated patterns
    if rotations:
        rotated = generated_rotations(pattern)

    # Randomly introduce the pattern into the landscape
    while patterns_introduced < number_of_patterns:

        if rotations:
            p = np.random.choice(rotated)
        else:
            p = pattern

        start_x = np.random.randint(0, high=size[0] - p.shape[0] + 1)
        start_y = np.random.randint(0, high=size[1] - p.shape[1] + 1)

        # Only introduce the pattern if the area is untouched
        if pattern_locations[start_x: start_x + p.shape[0], start_y: start_y + p.shape[1]].sum() == 0:
            # Write pattern
            landscape[start_x: start_x + p.shape[0], start_y: start_y + p.shape[1]] = p
            # Mark coordinates as touched
            pattern_locations[start_x: start_x + p.shape[0], start_y: start_y + p.shape[1]] = 1
            # Increase counter
            patterns_introduced += 1

        attempts += 1
        if attempts > 2 * number_of_patterns:
            raise StopIteration('The number of patterns was too high for the landscape size')

    return landscape


if __name__ == '__main__':

    import time
    start = time.time()

    bugs_ = find_pattern('landscape.txt', 'bug.txt', rotations=False)
    print(bugs_)

    landscape_ = generate_random_landscape((1000, 1000), 'bug.txt', 200, rotations=True)
    n = find_pattern(landscape_, 'bug.txt', rotations=True)
    print(n)

    end = time.time()
    print('{0:.2f}s'.format(end - start))
