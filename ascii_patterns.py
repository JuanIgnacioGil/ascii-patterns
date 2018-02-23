# -*- coding: utf-8 -*-
"""
Find the bug

Finds patters in ascii images

"""

import numpy as np


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
    list of tuples
        List of start coordinates (upper left) for each bug
    """

    if isinstance(landscape, str):
        landscape = matrix_from_file(landscape)

    if isinstance(pattern, str):
        pattern = matrix_from_file(pattern)

    # Look for the pattern
    bugs = []

    for i in range(landscape.shape[0] - pattern.shape[0] + 1):
        j = 0
        # Loop until the end of the row (a -2 indicates that everything that comes after is empty, so we stop looking)
        while (j <= landscape.shape[1] - pattern.shape[1]) and (landscape[i, j] != -2):

            if landscape[i, j] > -2:
                found_pattern = is_pattern(landscape, pattern, i, j)

                if found_pattern:
                    # Increase number of bugs
                    bugs.append((i, j))

                    # Delete the bug from the landscape, to accelerate the search
                    landscape[i:i + pattern.shape[0], j:j + pattern.shape[1]] = -3

                    # Advance j to the end of the pattern
                    j += pattern.shape[1] - 1

            j += 1

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
    for x in range(pattern.shape[0]):
        for y in range(pattern.shape[1]):
            if pattern[x, y] > 0:
                if landscape[i + x, j + y] != pattern[x, y]:
                    found_pattern = False
                    break
            elif pattern[x, y] == -2:  # Stop searching at the end of line
                break

        if found_pattern is False:
            break

    return found_pattern


def matrix_from_file(filename):
    """
    Reads a text file and returns a numpy array with a integer matrix representation

    The result is a numpy array of integers, where every character is represented by its ascii index, except:
    * White spaces at the start of a row are -1
    * White spaces at the end of a row are -2

    FIXME: (This won't properly if the pattern has white spaces at the end of the row which have to be matched)

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

        # Convert character to its ascii index for easier processing
        int_list = [ord(c) for c in char_list]
        raw_matrix.append(int_list)

    # All rows need the same number of elements. Insert a -2 at the end (we will use this -2 to match any character)
    # FIXME:
    # (This is probably not necessary (it comes from a previous version, where also rotated patterns
    # where matched, so the figured needed to be rectangular). Searching only until the real end of the line would
    # be more efficient)

    n_columns = max([len(r) for r in raw_matrix])
    min_columns = max([1, min([len(r) for r in raw_matrix])])  # Used to fill empty rows

    new_matrix = []

    for r in range(len(raw_matrix)):
        rc = len(raw_matrix[r])

        if 0 < rc < n_columns:
            new_matrix.append(raw_matrix[r] + [-2] * (n_columns - rc))
        elif rc == 0:
            # If the row is empty, we fill it withe spaces until the minimum figure length, and then with -2
            new_matrix.append([32] * min_columns + [-2] * (n_columns - min_columns))
        else:
            new_matrix.append(raw_matrix[r])

    matrix = np.vstack(new_matrix)

    for r in range(matrix.shape[0]):
        # Replace white spaces at the start of the row by a -1, so that they can be matched
        for k in range(n_columns):
            if matrix[r, k] == 32:
                matrix[r, k] = -1
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


def bugs_race(landscape_file, bug_east_file, bug_west_file):
    """

    Parameters
    ----------
    landscape_file: str
    bug_east: str
    bug_west: str

    Returns
    -------
    list of tuples

    """

    # Bugs east
    bugs_east = find_pattern(landscape_file, bug_east_file)
    bugs_west = find_pattern(landscape_file, bug_west_file)

    # Read patterns
    east_bug = matrix_from_file(bug_east_file)
    west_bug = matrix_from_file(bug_west_file)

    # Times for each eastern_bug to reach the end
    eastern_eta = [(90 - b[1], 'E') for b in bugs_east]

    # Times for each western_bug to reach the end
    western_eta = [(b[1] + west_bug.shape[1], 'W') for b in bugs_west]

    # Mix both
    eta = eastern_eta + western_eta

    # Sort
    sorted_eta = sorted(eta, key=lambda x: x[0])

    # Print the winner
    loser = sorted_eta[-1][1]
    winner = [x for x in ('E', 'W') if x is not loser]
    print('And the winner is ... {}!!!'.format(winner[0]))

    return sorted_eta


if __name__ == '__main__':

    import time
    start = time.time()

    # bugs_ = find_pattern('landscape.txt', 'bug.txt')
    # print(bugs_)
    #
    # bugs_ = find_pattern('landscape2.txt', 'bug2.txt')
    # print(bugs_)
    #
    # bugs_ = find_pattern('landscape2.txt', 'bug.txt')
    # print(bugs_)

    result = bugs_race('ApplicationTest-onsite/landscape.txt',
              'ApplicationTest-onsite/bug_east.txt',
              'ApplicationTest-onsite/bug_west.txt')

    print(result)

    end = time.time()
    print('{0:.2f}s'.format(end - start))
