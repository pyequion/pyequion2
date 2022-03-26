# PyEquion2

A pure python implementation for electrolytes chemical equilibrium.

Repository at https://github.com/pyequion/pyequion2

A heavily update version of the previous package found in https://github.com/caiofcm/pyequion

## Features

- Pure python package: hence it is easy to install in any platform
- Calculation of equilibrium of inorganic salts in water solution with precipitation and degassing
- Automatic determination of reactions
- Provides information as: Ionic Strength, pH, Electric Conductivity and the concentrations of each species as well their activity coefficient
- A modular approach for the activity coefficient calculation allows the usage of new thermodynamic models
- Calculation of equilibrium and reaction fluxes at reacting solid interfaces
- An incorporated GUI for ease-of-use

## Installation

The package can be installed with `pip install .` on this folder

You can also use PyPI, `pip install pyequion2`.

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