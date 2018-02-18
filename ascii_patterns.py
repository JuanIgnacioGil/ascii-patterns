# -*- coding: utf-8 -*-
"""
Find the bug

Finds patters in ascii images

"""

import numpy as np
from itertools import product


def find_pattern(landscape, pattern):
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

    Returns
    -------
    int
        Number of times the pattern appears in the image

    """
    if isinstance(landscape, str):
        landscape = matrix_from_file(landscape)

    if isinstance(pattern, str):
        pattern = matrix_from_file(pattern)

    # Look for the pattern
    bugs = 0

    for i, j in product(range(landscape.shape[0] - pattern.shape[0] + 1),
                        range(landscape.shape[1] - pattern.shape[1] + 1)):

        if landscape[i, j] >= -3:
            found_pattern = is_pattern(landscape, pattern, i, j)

            if found_pattern:
                # Increase number of bugs
                bugs += 1

                # Delete the bug from the landscape, to accelerate the search
                landscape[i:i + pattern.shape[0], j:j + pattern.shape[1]] = -3

    return bugs


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
        if pattern[x, y] > -2:
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

    # All rows need the same number of elements. Insert a -2 at the end (we will use this -2 to match any character)
    n_columns = max([len(r) for r in raw_matrix])
    min_columns = max([1, min([len(r) for r in raw_matrix])])  # Used to fill empty rows

    new_matrix = []

    for r in range(len(raw_matrix)):
        rc = len(raw_matrix[r])

        if 0 < rc < n_columns:
            new_matrix.append(raw_matrix[r] + [-2] * (n_columns - rc))
        elif rc == 0:
            # If the row is empty, we fill it withespaces until the minimum figure length, and then with -2
            new_matrix.append([32] * min_columns + [-2] * (n_columns - min_columns))
        else:
            new_matrix.append(raw_matrix[r])

    matrix = np.vstack(new_matrix)

    for r in range(matrix.shape[0]):
        # Replace white spaces at the start of the row by a -2, so that they can be matched
        for k in range(n_columns):
            if matrix[r, k] == 32:
                matrix[r, k] = -2
            else:
                break

    return matrix


def generate_random_landscape(size, pattern, number_of_patterns):
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
    pattern: str of numpy.array
        Text file for the pattern
    number_of_patterns: int
        Number of times we want to introduce the pattern into the landscape

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
    if isinstance(pattern, str):
        pattern = matrix_from_file(pattern)

    # Randomly introduce the pattern into the landscape
    while patterns_introduced < number_of_patterns:

        start_x = np.random.randint(0, high=size[0] - pattern.shape[0] + 1)
        start_y = np.random.randint(0, high=size[1] - pattern.shape[1] + 1)

        # Only introduce the pattern if the area is untouched
        if pattern_locations[start_x: start_x + pattern.shape[0], start_y: start_y + pattern.shape[1]].sum() == 0:
            # Write pattern
            landscape[start_x: start_x + pattern.shape[0], start_y: start_y + pattern.shape[1]] = pattern
            # Mark coordinates as touched
            pattern_locations[start_x: start_x + pattern.shape[0], start_y: start_y + pattern.shape[1]] = 1
            # Increase counter
            patterns_introduced += 1

        attempts += 1
        if attempts > 2 * number_of_patterns:
            raise StopIteration('The number of patterns was too high for the landscape size')

    return landscape


if __name__ == '__main__':

    import time
    start = time.time()

    bugs_ = find_pattern('landscape.txt', 'bug.txt')
    print(bugs_)

    bugs_ = find_pattern('landscape2.txt', 'bug2.txt')
    print(bugs_)

    bugs_ = find_pattern('landscape2.txt', 'bug.txt')
    print(bugs_)

    bug_ = np.random.randint(0, high=1000, size=(10, 5))
    landscape_ = generate_random_landscape((1000, 1000), bug_, 200)
    n = find_pattern(landscape_, bug_)
    print(n)

    landscape_ = generate_random_landscape((1000, 1000), 'bug.txt', 200)
    n = find_pattern(landscape_, 'bug.txt')
    print(n)

    landscape_ = generate_random_landscape((1000, 1000), 'bug2.txt', 200)
    n = find_pattern(landscape_, 'bug2.txt')
    print(n)

    end = time.time()
    print('{0:.2f}s'.format(end - start))
