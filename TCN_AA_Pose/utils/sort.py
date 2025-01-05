import re

def natural_sort_key(s:str):
    """
    Function to generate a natural sorting key for a string.
    This function splits the string into a list of numbers and non-numbers,
    and converts the numbers to integers for proper sorting.
    """
    return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]