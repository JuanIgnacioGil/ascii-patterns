# ascii_patterns

Search patterns in ascii images

## Use ##

### find_pattern ###
```python
find_pattern('landscape.txt', 'bug.txt')```

Finds the number of times the pattern in the file ```bug.txt``` can be found in the ascii
image in the file ```landscape.txt```.

Calling the function with the flag ```rotations=True``` also look for the rotated version of the pattern (for the monent, only 90 degrees rotations)

### generate_random_landscape ###

```python
generate_random_landscape((1000, 1000), 'bug.txt', 200, rotations=True)```
generates a numpy array of (1000, 1000) which has the pattern in file ```bug.txt``` 200 times, and which can be passed directly to ```find_pattern``` for testing purposes.

## Dependencies ##

The module has been tested with python 2.7.10. Apart from the standard python libraries, it uses numpy 1.14, because of how it facilitates working with matrices. It is also used for the random generation of landscapes.
