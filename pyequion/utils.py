# -*- coding: utf-8 -*-
from .builder import load_from_db, charge_number, stoich_number #For backward compatibility

# import copy
# import re

# import commentjson


# def load_from_db(fname):
#     if not isinstance(fname, str):
#         return copy.deepcopy(fname)
#     #Open a json file as dict
#     with open(fname, "r") as json_file:
#         db = commentjson.load(json_file)
#     return db


# def charge_number(specie):
#     """
#     Get charge number of specie
#     """
#     return 1*specie.count('+') - 1*specie.count('-')


# def stoich_number(specie, element):
#     """
#     Get stoichometric coeficient of element in specie
#     """
#     if element == 'e':  # Charge number
#         return charge_number(specie)
#     re_string = r"({0}[(A-Z)(\+\-)\d])|{0}$"
#     # If there is element, it will return either
#     # {0}{1}.format(element,char), or {0}.format(element)
#     # In second case, element is on the end, no number given,
#     # so stoich number is 1. In first case, if char is a int,
#     # stoich number is char, else, stoich number is 1
#     re_group = re.search(re_string.format(element), specie)
#     if re_group is None:  # No element in this specie
#         value = 0
#     else:
#         res_string = re_group.group(0)
#         if res_string == element:
#             value = 1
#         else:
#             try:
#                 value = int(res_string[-1])
#             except ValueError:
#                 value = 1
#     return value


# def _convert_to_extended_format(specie):
#     if "(" not in specie:
#         return specie
#     else:
#         return specie