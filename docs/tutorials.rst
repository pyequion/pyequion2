Tutorials
================================

Example 1 - Basic electrolytical balance calculation
----------------------------------------------------

Basic example for calculating electrolytical balance of a C, Ca, Na, Cl system.

First, we calculate the balance

.. code-block:: python
   :linenos:

   from pyequion2 import EquilibriumSystem #Import the necessary module
   eqsys = EquilibriumSystem(['CaCl2', 'NaHCO3']) #We set up the feed components of our system
   molal_balance = {'Ca':0.028, 'C':0.065, 'Na':0.065, 'Cl':0.056} #Set up the balances
   TK = 298.15 #Temperature in Kelvin
   PATM = 1.0 #Pressure in atm
   #Returns the solution class and tuple of solution statistics
   solution, solution_stats = eqsys.solve_equilibrium_mixed_balance(TK, molal_balance=molal_balance, PATM=PATM)

We go line by line here. First, we import the necessary module

>>> from pyequion2 import EquilibriumSystem

*EquilibriumSystem* is the main API interface in PyEquion. It is a general
class for setting up and calculating a variety of electrolytical balance systems.
Following examples will show the full extent of this class.

For now, we set up our system from the feed components

>>> eqsys = EquilibriumSystem(['CaCl2', 'NaHCO3'])

This sets up our system with all corresponding aqueous species involved in equilibrium.
We can inspect these

>>> eqsys.species
['H2O', 'CO2', 'CO3--', 'Ca++', 'CaCO3', 'CaCl2', 'CaHCO3+',
 'CaOH+', 'Cl-', 'H+', 'HCO3-', 'Na+', 'Na2CO3', 'NaCO3-',
 'NaHCO3', 'NaOH', 'OH-']
 
We also have the option of setting up our equilibrium system from the elements themselves

>>> eqsys = EquilibriumSystem(['Ca', 'Cl', 'Na', 'C'], from_elements=True)

By default, we use the Pitzer activity model. However, we can set up 
a different activity model at the class initialization

>>> eqsys = EquilibriumSystem(['CaCl2', 'NaHCO3'], activity_model="EXTENDED_DEBYE")

and change the activity model after we initialize the system

>>> eqsys.set_activity_functions(activity_model="PITZER")

After initializing our system, we set up the amount of feed components in molals

>>> molal_balance = {'Ca':0.028, 'C':0.065, 'Na':0.065, 'Cl':0.056}

Unlike other softwares, we here have to specify the amount of each element individually,
not following the feed components. This is intentional, for then we easily have the flexibility
of having non-stoichometric element balance.

>>> molal_balance = {'Ca':0.028, 'C':0.075, 'Na':0.065, 'Cl':0.056} #molal_balance['C'] != molal_balance['Na']

We are also free to specify in the molal balance the amount of specific ions in the water

>>> molal_balance = {'Ca++':0.028, 'C':0.075, 'Na':0.065, 'Cl':0.056} #Our solution will necessary have 0.028 molals of Ca++ ions.

To calculate our balance, we then specify both temperature and pressure

>>> TK = 298.15 #Temperature in Kelvin
>>> PATM = 1.0 #Pressure in atm

By default, if pressure is not provided, it is assumed that we have unit atmospheric pressure. The temperature must necessarily be provided.

We are then ready to calculate our electrolytical balance using the *solve_equilibrium_mixed_balance* method.

>>> solution, solution_stats = eqsys.solve_equilibrium_mixed_balance(TK, molal_balance=molal_balance, PATM=PATM)

We also return *solution_stats* for inspection of the performance of the solver.
Here, solution_stats is a dict with residual of the solution (solution_stats['res']),
that can be analyzed as an error metric. Also, solution stats has the 'x' attribute,
which is the numerical solution of molals (and other variables), that can be used as
initial guess for repeated calculation.

>>> import numpy as np
>>> residual = solution_stats['res']
>>> np.max(np.abs(solution_stats['res']))
7.513989430663059e-13

As for the *solution* object, it is an instance of *SolutionClass*, that contains 
varied information about the electrolytical balance of our system.

We can inspect our species balance, both in molals concentrations and their activities, 
through the attributes *solution.molals* and *solution.activities*

>>> for specie in solution.species:
...     print(solution.molals[specie], solution.activities[specie])
H2O 55.508435061791985 1.0
CO2 0.0017451987860942807 0.0017996296574881963
CO3-- 0.00029889095285012476 0.00010547453992485908
Ca++ 0.021336180473077816 0.007676960165455775
CaCO3 0.0013191134282240777 0.0013602551559961564
CaCl2 1.286269802134016e-05 1.3263871725652214e-05
CaHCO3+ 0.005331755169106101 0.004154442718456503
CaOH+ 8.823157066306216e-08 6.753529488353837e-08
Cl- 0.05597427460395732 0.041566218194101646
H+ 2.318795121973022e-08 1.8865072850393035e-08
HCO3- 0.05504929827427368 0.04242877615220498
Na+ 0.06374309497676348 0.04880999552873938
Na2CO3 1.1450565517797437e-06 1.1807696329515286e-06
NaCO3- 0.00012524189134245758 9.586419010815542e-05
NaHCO3 0.0011293564415574976 0.0011645798531929594
NaOH 1.6577233005177517e-08 1.7094259056857954e-08
OH- 7.258362733865141e-07 5.365250097984246e-07

Of course, we can also do the same thing for the amount of elements in our solution

>>> solution.elements_molals
{'H': 111.07838138730193,
 'O': 55.70169069365096,
 'Ca': 0.028,
 'Na': 0.06499999999999999,
 'Cl': 0.056,
 'C': 0.06499999999999999}

We can also inspect the saturation indexes of every possible solid precipitate

>>> solution.saturation_indexes
{'Aragonite': 2.2443969545735136,
 'Calcite': 2.388166801965749,
 'Halite': -4.262750719099147,
 'Vaterite': 1.8217353077514495}
 
Also, we have access to some properties of our system
 
>>> solution.ph
7.724341513364517
>>> solution.ionic_strength #mol/kg H2O
0.13338239393747503
>>> solution.electrical_conductivity #S/m
0.8269708424123452

Finally, we can generate a log file of our solution, for external inspection

>>> solution.savelog("solutionlog.txt")
>>> with open("solutionlog.txt", "r") as f:
...     print(f.read())
CONDITIONS
aqueous
T - 298.15 K
P - 1.00 atm
[Ca]=2.80e-02 mol/kg H2O
[C]=6.50e-02 mol/kg H2O
[Na]=6.50e-02 mol/kg H2O
[Cl]=5.60e-02 mol/kg H2O
----------------------------------------
[COMPONENT]    [CONCENTRATION (mol/kg H2O)]    [ACTIVITY (mol/kg H2O)]    [MOLE FRACTION]
H2O    5.551e+01    1.000e+00    9.963014e-01
Na+    6.374e-02    4.881e-02    1.144102e-03
Cl-    5.597e-02    4.157e-02    1.004663e-03
HCO3-    5.505e-02    4.243e-02    9.880605e-04
Ca++    2.134e-02    7.677e-03    3.829556e-04
CaHCO3+    5.332e-03    4.154e-03    9.569780e-05
CO2    1.745e-03    1.800e-03    3.132396e-05
CaCO3    1.319e-03    1.360e-03    2.367630e-05
NaHCO3    1.129e-03    1.165e-03    2.027042e-05
CO3--    2.989e-04    1.055e-04    5.364689e-06
NaCO3-    1.252e-04    9.586e-05    2.247923e-06
CaCl2    1.286e-05    1.326e-05    2.308680e-07
Na2CO3    1.145e-06    1.181e-06    2.055222e-08
OH-    7.258e-07    5.365e-07    1.302778e-08
CaOH+    8.823e-08    6.754e-08    1.583637e-09
H+    2.319e-08    1.887e-08    4.161924e-10
NaOH    1.658e-08    1.709e-08    2.975389e-10
C    6.500e-02
Ca    2.800e-02
Cl    5.600e-02
H    1.111e+02
Na    6.500e-02
O    5.570e+01
----------------------------------------
PROPERTIES
pH = 7.724
I = 0.133 mol/kg H2O
conductivity = 0.827 S/m
----------------------------------------
[PHASE]    [AMOUNT mol/kg H2O]
Aragonite    0.0
Calcite    0.0
Halite    0.0
Vaterite    0.0
CO2(g)    0.0
H2O(g)    0.0
----------------------------------------
[PHASE]    [SUPERSATURATION]    [SI]
Aragonite    175.54843195012336   2.2443969545735136
Calcite    244.43691952856932   2.388166801965749
Halite    5.4607121083262775e-05   -4.262750719099147
Vaterite    66.33386579002101   1.8217353077514495

You can forget for now the [PHASE] [AMOUNT] sub-block,
it will be explained later in another example.


Example 2 - Electrolytical balance with fixed pH
------------------------------------------------

Consider the following setting: we add to some aqueous solution a concentration of 
150 mmolal of NaHCO3. We then let the solution in contact with the air (open system), 
at 35 ºC, and, after some time, we measure the pH of the solution, and find the value 9.0. 
We want to then know the concentration of carbon left in our system.

We first set up our system.

>>> from pyequion2 import EquilibriumSystem
>>> EquilibriumSystem(['Na', 'C'], from_elements=True)

Since there is no volatile Na component, we know that it's amount must be 
the same that we've put in water.

>>> molal_balance = {'Na': 0.150}

Now, we must then fix our pH. Knowing that the pH is by definition the negative of 
the base 10 logarithm of H+ activity, then we fix that in our calculation

>>> activities_balance_log = {'H+': -9.0} #Fix our pH by fixing H+ log-activity

Finally, fix our temperature in Kelvin

>>> TK = 35 + 273.15 #35 ºC to Kelvin

We are now ready to calculate the electrolytical balance

>>> solution, solution_stats = \
...           eqsys.solve_equilibrium_mixed_balance(TK,
...                                                 molal_balance=molal_balance,
...                                                 activities_balance_log=activities_balance_log)

Inspect the residual of our solution

>>> import numpy as np
>>> np.max(np.abs(solution_stats['res']))
8.505418591653324e-13

Check the pH of our solution is correct

>>> solution.ph
9.000000000000147

Retrieve the total amount of carbon in our system

>>> solution.elements_molals['C']
0.12677211939028027

Example 3 - Equilibrium with precipitation
------------------------------------------

Let's set up a similar case than in example 1

>>> from pyequion2 import EquilibriumSystem #Import the necessary module
>>> eqsys = EquilibriumSystem(['Ca', 'Na', 'C', 'Cl'], from_elements=True)
>>> molal_balance = {'Ca':0.028, 'C':0.065, 'Na':0.065, 'Cl':0.056}
>>> TK = 298.15 #Temperature in Kelvin
>>> PATM = 1.0 #Pressure in atm

Now we change things a little. Instead of calculating the supersaturation of solid species, 
we assume that the most stable phase of each possible solid precipitates, if possible, 
and calculates the amount of precipitate.

>>> solution, solution_stats = eqsys.solve_equilibrium_elements_balance_phases(TK, molal_balance)

Let's check the precipitating phases

>>> solution.solid_molals
{'Aragonite': 0.0,
 'Calcite': 0.023947112655582508,
 'Halite': 3.507133834338337e-116,
 'Vaterite': 0.0}
 
Unit here is important. The value of precipitated calcite here are 
the mols of calcite per unit of liquid H2O.

We can check that there is no degassing in our system

>>> solution.gas_molals
{'CO2(g)': 1.0714181266404203e-26, 'H2O(g)': 0.0}


Of course, the amount of dissolved carbon and calcium in water can't be conserved.
 
>>> solution.elements_molals['Ca'], solution.elements_molals['C']
(0.0370571521476301, 5.7152147630119065e-05)

But considering also the solid precipitation, we again have conservation

>>> solution.solid_molals['Calcite'] + solution.elements_molals['Ca']
0.028
>>> solution.solid_molals['Calcite'] + solution.elements_molals['C']
0.06499999999999997

Now we are in a position to explain what the [PHASE] [AMOUNT] block 
of the log means. It is simply the amount of precipitated phase 
for each possible precipitate.

Example 4 - CO2 degassing
-------------------------

We can also study degassing for our system. Consider a very simple system

>>> from pyequion2 import EquilibriumSystem #Import the necessary module
>>> eqsys = EquilibriumSystem(['CO2']) #We set up the feed components of our system
>>> molal_balance = {'C':0.5} #Set up the balances
>>> TK = 298.15
>>> solution, _ = eqsys.solve_equilibrium_elements_balance_phases(TK, molal_balance)

We can see that we see a lot of degassing in this setting

>>> solution.gas_molals['CO2(g)']
0.49972244867277366

Of course, by increasing pressure, we can make all of the CO2 gas dissolved again

>>> PATM = 100.0
>>> solution_high_pressure, _ = eqsys.solve_equilibrium_elements_balance_phases(TK,
...                                                                             molal_balance,
...                                                                             PATM=PATM)
>>> solution_high_pressure.gas_molals['CO2(g)']
1e-200