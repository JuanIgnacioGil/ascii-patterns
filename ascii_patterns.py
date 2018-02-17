# -*- coding: utf-8 -*-
"""
Find the bug

Finds patters in ascii images
"""

import numpy as np
from itertools import product


def find_pattern(landscape_file, bug_file):
    """
    Count the number of times a pattern (read from a txt file) appers in an ascii image (read from another file)

    Parameters
    ----------
    landscape_file: str
        Path of the ascii image
    bug_file: str
        Path of the pattern

    Returns
    -------
    int
        Number of times the pattern appears in the image

    """

    landscape = matrix_from_file(landscape_file)
    pattern = matrix_from_file(bug_file)

    bugs = 0

    # Look for the pattern
    for i, j in product(range(landscape.shape[0] - pattern.shape[0] + 1),
                        range(landscape.shape[1] - pattern.shape[1] + 1)):

        # Look for the pattern
        found_pattern = is_pattern(landscape, pattern, i, j)

        if found_pattern:
            # Increase number of bugs
            bugs += 1

            # Delete the bug from the landscape, to accelerate the search
            landscape[i:i + pattern.shape[0], j:j + pattern.shape[1]] = -1

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

    # All rows need the same number of elements. Insert whitespaces (32) at the end
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


if __name__ == '__main__':
    bugs_ = find_pattern('landscape.txt', 'bug.txt')
    print(bugs_)
