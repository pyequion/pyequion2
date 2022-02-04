# -*- coding: utf-8 -*-
import os
import pathlib

ownpath = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
filepath = ownpath/'pitzer.txt'
with open(filepath, 'r') as file:
    text1 = file.read()

import pitzer_data
text2 = pitzer_data.pitzer_data