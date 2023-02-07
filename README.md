# PyEquion2

A pure python implementation for electrolytes chemical equilibrium.

Repository at https://github.com/pyequion/pyequion2

A heavily update version of the previous package found in https://github.com/caiofcm/pyequion

Tests of the accuracy of its parameters and equations are underway, so results may change. Its API may change, and modules and functions may be added, removed and renamed. Use at your own peril!

## Features

- Pure python package: hence it is easy to install in any platform
- Calculation of equilibrium of inorganic salts in water solution with precipitation and degassing
- Automatic determination of reactions
- Provides information as: Ionic Strength, pH, Electric Conductivity and the concentrations of each species as well their activity coefficient
- A modular approach for the activity coefficient calculation allows the usage of new thermodynamic models
- Calculation of equilibrium and reaction fluxes at reacting solid interfaces
- An incorporated GUI for ease-of-use

## Installation

The package can be installed either running downloading the 
source code and `pip install .` on this folder, or directly by

```
pip install git+https://github.com/pyequion/pyequion2
```

There is an older PyPI version, but is not supported anymore


## GUI

To run a GUI version of PyEquion2, just run

```
import pyequion2
pyequion2.rungui()
```


## Contributors

- Caio Curitiba Marcellos
- Danilo Naiff
- Gerson Francisco da Silva Junior
- Elvis do Amaral Soares
- Fabio Ramos
- Amaro G. Barreto Jr
