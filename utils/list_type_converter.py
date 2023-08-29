def convert_to_type(string_list, convert_to = int):
    # initialize an empty list to store the converted integers
    int_list = []
    # loop through each element in the string list
    for s in string_list:
        # try to convert the element to an integer and append it to the int list
        try:
            i = convert_to(s)
            int_list.append(i)
        # if the element cannot be converted, raise an exception and exit the function
        except ValueError:
            print(f"Invalid input: {s} is not a valid {convert_to.__name__}")
            return None
    # return the int list as the output
    return int_list


import numpy as np # import the numpy module

def convert_to_array(list_of_lists):
    # use the np.array() function to convert the list of lists into a numpy array
    array = np.array(list_of_lists)
    # return the array as the output
    return array
