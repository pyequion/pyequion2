# -*- coding: utf-8 -*-

pitzer_data = \
r"""PITZER
-B0
  B(OH)4-     K+      0.035
  B(OH)4-     Na+    -0.0427
  B3O3(OH)4-  K+     -0.13
  B3O3(OH)4-  Na+    -0.056
  B4O5(OH)4-2 K+     -0.022
  B4O5(OH)4-2 Na+    -0.11
  Ba+2      Br-       0.31455       0           0          -0.33825E-3
  Ba+2      Cl-       0.5268        0           0           0         0     4.75e4  # ref. 3
  Ba+2      OH-       0.17175
  Br-       H+        0.1960        0           0          -2.049E-4
  Br-       K+        0.0569        0           0           7.39E-4
  Br-       Li+       0.1748        0           0          -1.819E-4
  Br-       Mg+2      0.4327        0           0          -5.625E-5
  Br-       Na+       0.0973        0           0           7.692E-4
  Br-       Sr+2      0.331125      0           0          -0.32775E-3
  Ca+2      Br-       0.3816        0           0          -5.2275E-4
  Ca+2      Cl-       0.3159        0           0          -3.27e-4   1.4e-7       # ref. 3
  Ca+2      HCO3-     0.4
  Ca+2      HSO4-     0.2145
  Ca+2      OH-      -0.1747
  Ca+2      SO4-2     0      # ref. 3
  CaB(OH)4+   Cl-     0.12
  Cl-       Fe+2      0.335925
  Cl-       H+        0.1775        0           0          -3.081E-4
  Cl-       K+        0.04808    -758.48       -4.7062      0.010072   -3.7599e-6  # ref. 3
  Cl-       Li+       0.1494        0           0          -1.685E-4
  Cl-       Mg+2      0.351         0           0          -9.32e-4   5.94e-7      # ref. 3
  Cl-       MgB(OH)4+  0.16
  Cl-       MgOH+    -0.1
  Cl-       Mn+2      0.327225
  Cl-       Na+       7.534e-2   9598.4        35.48     -5.8731e-2   1.798e-5   -5e5  # ref. 3
  Cl-       Sr+2      0.2858        0           0           0.717E-3
  CO3-2     K+        0.1488        0           0           1.788E-3
  CO3-2     Na+       0.0399        0           0           1.79E-3
  Fe+2      HSO4-     0.4273
  Fe+2      SO4-2     0.2568
  H+        HSO4-     0.2065
  H+        SO4-2     0.0298
  HCO3-     K+        0.0296        0           0           0.996E-3
  HCO3-     Mg+2      0.329
  HCO3-     Na+      -0.018  # ref. 3 + new -analytic for calcite
  HCO3-     Sr+2      0.12
  HSO4-     K+       -0.0003
  HSO4-     Mg+2      0.4746
  HSO4-     Na+       0.0454
  K+        OH-       0.1298
  K+        SO4-2     3.17e-2       0           0           9.28e-4               # ref. 3
  Li+       OH-       0.015
  Li+       SO4-2     0.136275      0           0           0.5055E-3
  Mg+2      SO4-2     0.2135     -951           0          -2.34e-2   2.28e-5     # ref. 3
  Mn+2      SO4-2     0.2065
  Na+       OH-       0.0864        0           0           7.00E-4
  Na+       SO4-2     2.73e-2      0          -5.8         9.89e-3  0          -1.563e5 # ref. 3
  SO4-2     Sr+2      0.200         0           0          -2.9E-3
-B1
  B(OH)4-   K+        0.14
  B(OH)4-   Na+       0.089
  B3O3(OH)4-  Na+    -0.910
  B4O5(OH)4-2 Na+    -0.40
  Ba+2      Br-       1.56975       0           0           6.78E-3
  Ba+2      Cl-       0.687         0           0           1.417e-2              # ref. 3
  Ba+2      OH-       1.2
  Br-       H+        0.3564        0           0           4.467E-4
  Br-       K+        0.2212        0           0           17.40E-4
  Br-       Li+       0.2547        0           0           6.636E-4
  Br-       Mg+2      1.753         0           0           3.8625E-3
  Br-       Na+       0.2791        0           0           10.79E-4
  Br-       Sr+2      1.7115        0           0           6.5325E-3
  Ca+2      Br-       1.613         0           0           6.0375E-3
  Ca+2      Cl-       1.614         0           0           7.63e-3    -8.19e-7   # ref. 3
  Ca+2      HCO3-     2.977 # ref. 3 + new -analytic for calcite
  Ca+2      HSO4-     2.53
  Ca+2      OH-      -0.2303
  Ca+2      SO4-2     3.546         0           0           5.77e-3               # ref. 3
  Cl-       Fe+2      1.53225
  Cl-       H+        0.2945        0           0           1.419E-4
  Cl-       K+        0.2168        0          -6.895       2.262e-2   -9.293e-6  -1e5  # ref. 3
  Cl-       Li+       0.3074        0           0           5.366E-4
  Cl-       Mg+2      1.65          0           0          -1.09e-2     2.60e-5   # ref. 3
  Cl-       MgOH+     1.658
  Cl-       Mn+2      1.55025
  Cl-       Na+       0.2769        1.377e4    46.8        -6.9512e-2   2e-5      -7.4823e5  # ref. 3
  Cl-       Sr+2      1.667         0           0           2.8425E-3
  CO3-2     K+        1.43          0           0           2.051E-3
  CO3-2     Na+       1.389         0           0           2.05E-3
  Fe+2      HSO4-     3.48
  Fe+2      SO4-2     3.063
  H+        HSO4-     0.5556
  HCO3-     K+        0.25          0           0           1.104E-3              # ref. 3
  HCO3-     Mg+2      0.6072
  HCO3-     Na+       0     # ref. 3 + new -analytic for calcite
  HSO4-     K+        0.1735
  HSO4-     Mg+2      1.729
  HSO4-     Na+       0.398
  K+        OH-       0.32
  K+        SO4-2     0.756        -1.514e4   -80.3         0.1091                # ref. 3
  Li+       OH-       0.14
  Li+       SO4-2     1.2705        0           0           1.41E-3
  Mg+2      SO4-2     3.367        -5.78e3      0          -1.48e-1   1.576e-4    # ref. 3
  Mn+2      SO4-2     2.9511
  Na+       OH-       0.253         0           0           1.34E-4
  Na+       SO4-2     0.956         2.663e3     0           1.158e-2   0         -3.194e5   # ref. 3
  SO4-2     Sr+2      3.1973        0           0          27e-3
-B2
  Ca+2      Cl-      -1.13          0           0        -0.0476                  # ref. 3
  Ca+2      OH-      -5.72
  Ca+2      SO4-2   -59.3           0           0        -0.443       -3.96e-6    # ref. 3
  Fe+2      SO4-2   -42.0
  HCO3-     Na+       8.22          0           0        -0.049                   # ref. 3 + new -analytic for calcite
  Mg+2      SO4-2   -32.45          0          -3.236e3  21.812       -1.8859e-2  # ref. 3
  Mn+2      SO4-2   -40.0
  SO4-2     Sr+2    -54.24          0           0          -0.42
-C0
  B(OH)4-   Na+       0.0114
  Ba+2      Br-      -0.0159576
  Ba+2      Cl-      -0.143      -114.5  # ref. 3
  Br-       Ca+2     -0.00257
  Br-       H+        0.00827       0           0        -5.685E-5
  Br-       K+       -0.00180       0           0        -7.004E-5
  Br-       Li+       0.0053        0           0        -2.813E-5
  Br-       Mg+2      0.00312
  Br-       Na+       0.00116       0           0        -9.30E-5
  Br-       Sr+2      0.00122506
  Ca+2      Cl-       1.4e-4      -57          -0.098    -7.83e-4      7.18e-7    # ref. 3
  Ca+2      SO4-2     0.114  # ref. 3
  Cl-       Fe+2     -0.00860725
  Cl-       H+        0.0008        0           0         6.213E-5
  Cl-       K+       -7.88e-4     91.27        0.58643  -1.298e-3     4.9567e-7  # ref. 3
  Cl-       Li+       0.00359       0           0        -4.520E-5
  Cl-       Mg+2      0.00651       0  0       -2.50e-4   2.418e-7                # ref. 3
  Cl-       Mn+2     -0.0204972
  Cl-       Na+       1.48e-3    -120.5        -0.2081    0            1.166e-7  11121  # ref. 3
  Cl-       Sr+2     -0.00130
  CO3-2     K+       -0.0015
  CO3-2     Na+       0.0044
  Fe+2      SO4-2     0.0209
  H+        SO4-2     0.0438
  HCO3-     K+       -0.008
  K+        OH-       0.0041
  K+        SO4-2     8.18e-3    -625          -3.30      4.06e-3                 # ref. 3
  Li+       SO4-2    -0.00399338    0           0        -2.33345e-4
  Mg+2      SO4-2     2.875e-2      0          -2.084     1.1428e-2   -8.228e-6   # ref. 3
  Mn+2      SO4-2     0.01636
  Na+       OH-       0.0044        0           0       -18.94E-5
  Na+       SO4-2     3.418e-3   -384           0        -8.451e-4     0        5.177e4  # ref. 3
-THETA
  B(OH)4-   Cl-      -0.065
  B(OH)4-   SO4-2    -0.012
  B3O3(OH)4-  Cl-     0.12
  B3O3(OH)4-  HCO3-  -0.10
  B3O3(OH)4-  SO4-2   0.10
  B4O5(OH)4-2  Cl-    0.074
  B4O5(OH)4-2  HCO3- -0.087
  B4O5(OH)4-2  SO4-2  0.12
  Ba+2      Na+       0.07   # ref. 3
  Br-       OH-      -0.065
  Ca+2      H+        0.092
  Ca+2      K+       -5.35e-3       0           0         3.08e-4               # ref. 3
  Ca+2      Mg+2      0.007
  Ca+2      Na+       9.22e-2       0           0        -4.29e-4      1.21e-6  # ref. 3
  Cl-       CO3-2    -0.02
  Cl-       HCO3-     0.03
  Cl-       HSO4-    -0.006
  Cl-       OH-      -0.05
  Cl-       SO4-2     0.03   # ref. 3
  CO3-2     OH-       0.1
  CO3-2     SO4-2     0.02
  H+        K+        0.005
  H+        Mg+2      0.1
  H+        Na+       0.036
  HCO3-     CO3-2    -0.04
  HCO3-     SO4-2     0.01
  K+        Na+      -0.012
  Mg+2      Na+       0.07
  Na+       Sr+2      0.051
  OH-       SO4-2    -0.013
-LAMDA
  B(OH)3    Cl-       0.091
  B(OH)3    K+       -0.14
  B(OH)3    Na+      -0.097
  B(OH)3    SO4-2     0.018
  B3O3(OH)4-  B(OH)3 -0.20
  Ca+2      CO2       0.183
  Ca+2      H4SiO4    0.238    # ref. 3
  Cl-       CO2      -0.005
  CO2       CO2      -1.34e-2   348   0.803 # new VM("CO2"), CO2 solubilities at high P, 0 - 150ï¿½C
  CO2       HSO4-    -0.003
  CO2       K+        0.051
  CO2       Mg+2      0.183
  CO2       Na+       0.085
  CO2       SO4-2     0.075    # Rumpf and Maurer, 1993.
  H4SiO4    K+        0.0298   # ref. 3
  H4SiO4    Li+       0.143    # ref. 3
  H4SiO4    Mg+2      0.238  -1788   -9.023  0.0103    # ref. 3
  H4SiO4    Na+       0.0566    75.3  0.115            # ref. 3
  H4SiO4    SO4-2    -0.085      0    0.28  -8.25e-4   # ref. 3
-ZETA
  B(OH)3    Cl-       H+        -0.0102
  B(OH)3    Na+       SO4-2      0.046
  Cl-       H4SiO4    K+        -0.0153  # ref. 3
  Cl-       H4SiO4    Li+       -0.0196  # ref. 3
  CO2       Na+       SO4-2     -0.015
-PSI
  B(OH)4-     Cl-     Na+       -0.0073
  B3O3(OH)4-  Cl-     Na+       -0.024
  B4O5(OH)4-2 Cl-     Na+        0.026
  Br-       K+        Na+       -0.0022
  Br-       K+        OH-       -0.014
  Br-       Na+       H+        -0.012
  Br-       Na+       OH-       -0.018
  Ca+2      Cl-       H+        -0.015
  Ca+2      Cl-       K+        -0.025
  Ca+2      Cl-       Mg+2      -0.012
  Ca+2      Cl-       Na+       -1.48e-2  0    0  -5.2e-6       # ref. 3
  Ca+2      Cl-       OH-       -0.025
  Ca+2      Cl-       SO4-2     -0.122    0    0  -1.21e-3      # ref. 3
  Ca+2      K+        SO4-2     -0.0365                         # ref. 3
  Ca+2      Mg+2      SO4-2      0.024
  Ca+2      Na+       SO4-2     -0.055  17.2                    # ref. 3
  Cl-       Br-       K+         0
  Cl-       CO3-2     K+         0.004
  Cl-       CO3-2     Na+        0.0085
  Cl-       H+        K+        -0.011
  Cl-       H+        Mg+2      -0.011
  Cl-       H+        Na+       -0.004
  Cl-       HCO3-     Mg+2      -0.096
  Cl-       HCO3-     Na+        0                              # ref. 3 + new -analytic for calcite
  Cl-       HSO4-     H+         0.013
  Cl-       HSO4-     Na+       -0.006
  Cl-       K+        Mg+2      -0.022  -14.27                  # ref. 3
  Cl-       K+        Na+       -0.0015   0    0   1.8e-5       # ref. 3
  Cl-       K+        OH-       -0.006
  Cl-       K+        SO4-2     -1e-3                           # ref. 3
  Cl-       Mg+2      MgOH+      0.028
  Cl-       Mg+2      Na+       -0.012   -9.51 # ref. 3
  Cl-       Mg+2      SO4-2     -0.008   32.63 # ref. 3
  Cl-       Na+       OH-       -0.006
  Cl-       Na+       SO4-2      0             # ref. 3
  Cl-       Na+       Sr+2      -0.0021
  CO3-2     HCO3-     K+         0.012
  CO3-2     HCO3-     Na+        0.002
  CO3-2     K+        Na+        0.003
  CO3-2     K+        OH-       -0.01
  CO3-2     K+        SO4-2     -0.009
  CO3-2     Na+       OH-       -0.017
  CO3-2     Na+       SO4-2     -0.005
  H+        HSO4-     K+        -0.0265
  H+        HSO4-     Mg+2      -0.0178
  H+        HSO4-     Na+       -0.0129
  H+        K+        Br-       -0.021
  H+        K+        SO4-2      0.197
  HCO3-     K+        Na+       -0.003
  HCO3-     Mg+2      SO4-2     -0.161
  HCO3-     Na+       SO4-2     -0.005
  HSO4-     K+        SO4-2     -0.0677
  HSO4-     Mg+2      SO4-2     -0.0425
  HSO4-     Na+       SO4-2     -0.0094
  K+        Mg+2      SO4-2     -0.048
  K+        Na+       SO4-2     -0.010
  K+        OH-       SO4-2     -0.050
  Mg+2      Na+       SO4-2     -0.015
  Na+       OH-       SO4-2     -0.009"""