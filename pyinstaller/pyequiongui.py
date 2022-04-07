# -*- coding: utf-8 -*-
import pyequion2

try:
    pyequion2.rungui()
except Exception as e:
    with open("ERRORLOG", "a") as f:
        f.write(str(e))
        f.write("\n")
