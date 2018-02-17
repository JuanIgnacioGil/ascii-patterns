# -*- coding: utf-8 -*-
"""
Find the bug

Finds patters in ascii images
"""

import numpy as np


def find_bug(landscape_file, bug_file):
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

    print(np.abs(pattern).sum())

    # Look for first file of the pattern
    candidates = [l for l in range(len(landscape)) if pattern[0] in landscape[l]]
    print(candidates)

    bugs = 0

    return bugs


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
    bugs_ = find_bug('landscape.txt', 'bug.txt')
    print(bugs_)
