import numpy as np
def find_list_index(lists_of_tradables, dynamic_param_list):
  """
  Finds the index of the list in the nested list that matches the string.

  Args:
    nested_list: The nested list.
    string: The list of strings to match.

  Returns:
    A list of the indices of the list in the nested list that matches the string.
  """
  indices = []
  for i, list_of_tradables in enumerate(lists_of_tradables):
    # if any(s in sublist for s in string_list):
    #   indices.append(i)
    # print(f"({i}) -> Checking if {dynamic_param_list} is in {list_of_tradables}")
    if len(np.shape(dynamic_param_list)) == 1:
      if any(s in list_of_tradables for s in dynamic_param_list):
        indices.append(i)
    else:
      for j, s in enumerate(list_of_tradables):
          # print(f"({j}) -> Checking if {s} is in {string_list}")
          if len(np.shape(dynamic_param_list)) == 2:
            if s in dynamic_param_list:
              indices.append(j)
          else:
            if s == dynamic_param_list[0]:
              indices.append(j)
  return indices   
