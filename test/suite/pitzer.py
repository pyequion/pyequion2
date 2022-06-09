# -*- coding: utf-8 -*-
import numpy as np

import reaktoro
import pyequion2

def convert(s):
    convert_map = {'CO3-2': 'CO3--',
                   'Ca+2': 'Ca++',
                   'Mg+2': 'Mg++',
                   'Fe+2': 'Fe++'}
    return convert_map.get(s, s)

class Tests(object):
    def test1(c=1.0, TK=298.15): #OK
        #B0, B1, C0
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O Na+ Cl-")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        res1 = float(rkprops.speciesActivityCoefficient("Na+"))
        res2 = float(rkprops.speciesActivityCoefficient("Cl-"))
        solutes = ['Na+', 'Cl-']
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([c, c]), TK)
        res3 = acts[1]
        res4 = acts[2]
        rel_error1 = np.abs(res1 - res3)/np.abs(res1)
        rel_error2 = np.abs(res2 - res4)/np.abs(res2)
        return rel_error1, rel_error2
    
    def test2(c=1.0, cco2=1.0, TK=298.15): #OK (CO2 differs but no CO2-CO2 reaktoro)
        #B0, B1, C0, LAMBDA
        solutes = ['Na+', 'Cl-', 'CO2']
        print(solutes)
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O Na+ Cl- CO2")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        res1a = float(rkprops.speciesActivityCoefficient("Na+"))
        res2a = float(rkprops.speciesActivityCoefficient("Cl-"))
        res3a = float(rkprops.speciesActivityCoefficient("CO2"))
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([c, c, cco2]), TK)
        res1b = acts[1]
        res2b = acts[2]
        res3b = acts[3]
        relerr1 = np.abs(res1a - res1b)/np.abs(res1a)
        relerr2 = np.abs(res2a - res2b)/np.abs(res2a)
        relerr3 = np.abs(res3a - res3b)/np.abs(res3a)
        return relerr1, relerr2, relerr3
    
    def test3(c=1.0, TK=298.15): #OK
        #B0, B1, B2, C0
        print(["Ca+2", "Cl-"])
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O Ca+2 Cl-")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", c, "mol")
        rkstate.set("Cl-", c, "mol")
        #reaktoro.equilibrate(rkstate)
        rkprops = reaktoro.ChemicalProps(rkstate)
        res1a = float(rkprops.speciesActivityCoefficient("Ca+2"))
        res2a = float(rkprops.speciesActivityCoefficient("Cl-"))
        solutes = ['Ca++', 'Cl-']
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([c, c]), TK)
        res1b = acts[1]
        res2b = acts[2]
        relerr1 = np.abs(res1a - res1b)/np.abs(res1a)
        relerr2 = np.abs(res2a - res2b)/np.abs(res2a)
        return relerr1, relerr2
    
    def test4(c=1.0, cco2=1.0, TK=298.15): #Not OK
        print(["Ca+2", "Cl-", "CO2"])
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O Ca+2 Cl- CO2")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        #reaktoro.equilibrate(rkstate)
        rkprops = reaktoro.ChemicalProps(rkstate)
        res1a = float(rkprops.speciesActivityCoefficient("Ca+2"))
        res2a = float(rkprops.speciesActivityCoefficient("Cl-"))
        res3a = float(rkprops.speciesActivityCoefficient("CO2"))
        solutes = ['Ca++', 'Cl-', 'CO2']
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([c, c, cco2]), TK)
        res1b = acts[1]
        res2b = acts[2]
        res3b = acts[3]
        relerr1 = np.abs(res1a - res1b)/np.abs(res1a)
        relerr2 = np.abs(res2a - res2b)/np.abs(res2a)
        relerr3 = np.abs(res3a - res3b)/np.abs(res3a)
        return relerr1, relerr2, relerr3
    
    def test5(ca=1, cb=1.4*1e-7, TK=298.15): #OK
        #B0, B1, C0
        print(["Na+", "Cl-", "H+", "OH-"])
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O Na+ Cl- H+ OH-")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", ca, "mol")
        rkstate.set("Cl-", ca, "mol")
        rkstate.set("H+", cb, "mol")
        rkstate.set("OH-", cb, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        res1a = float(rkprops.speciesActivityCoefficient("Na+"))
        res2a = float(rkprops.speciesActivityCoefficient("Cl-"))
        res3a = float(rkprops.speciesActivityCoefficient("H+"))
        res4a = float(rkprops.speciesActivityCoefficient("OH-"))
        solutes = ['Na+', 'Cl-', 'H+', 'OH-']
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([ca, ca, cb, cb]), TK)
        res1b = acts[1]
        res2b = acts[2]
        res3b = acts[3]
        res4b = acts[4]
        relerr1 = np.abs(res1a - res1b)/np.abs(res1a)
        relerr2 = np.abs(res2a - res2b)/np.abs(res2a)
        relerr3 = np.abs(res3a - res3b)/np.abs(res3a)
        relerr4 = np.abs(res4a - res4b)/np.abs(res4a)
        return relerr1, relerr2, relerr3, relerr4
    
    def test6(c=1.0, TK=298.15):
        #B0, B1, C0
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        rksolution = reaktoro.AqueousPhase("H2O H+ OH-")
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("H+", c, "mol")
        rkstate.set("OH-", c, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        res3a = float(rkprops.speciesActivityCoefficient("H+"))
        res4a = float(rkprops.speciesActivityCoefficient("OH-"))
        solutes = ['H+', 'OH-']
        model = pyequion2.activity.setup_pitzer(solutes)
        acts = 10**model(np.array([c, c]), TK)
        res3b = acts[1]
        res4b = acts[2]
        relerr3 = np.abs(res3a - res3b)/np.abs(res3a)
        relerr4 = np.abs(res4a - res4b)/np.abs(res4a)
        return relerr3, relerr4
    
    def test7(c=1.0, cco2=1e-2, ch=1e-3, TK=298.15): #Not OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Ca+2', 'Cl-', 'CO2', 'H+', 'OH-']
        print(specs)
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", 2*c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, c, cco2, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test8(c=1.0, cco2=1e-2, ch=1e-3, TK=298.15): #OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Na+', 'Cl-', 'CO2', 'H+', 'OH-']
        print(specs)
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        pq_array = np.array([c, c, cco2, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test9(c=1.0, cco2=1e-2, ch=1e-3, TK=298.15): #OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Cl-', 'CO2', 'H+', 'OH-']
        print(specs)
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        #rkstate.set("Ca+2", 2*c, "mol")
        rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, cco2, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test10(c=1.0, cco2=1e-2, ch=1e-3, TK=298.15):
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Ca+2', 'CO2', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("CO2", cco2, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, cco2, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test11(c=1.0, ch=1e-3, TK=298.15): #Not OK (H+)
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Ca+2', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test12(c=1.0, ch=1e-3, TK=298.15): #OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Ca+2', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Ca+2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test13(c=1.0, ch=1e-3, TK=298.15): #OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Na+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test14(c=1.0, ch=1e-3, TK=298.15): #OK
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Na+', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Na+", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            print(np.log(resa), np.log(resb), spec)
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test15(c=1.0, ch=1e-3, TK=298.15): #Not OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Mg+2', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Mg+2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch*nullifier, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch*nullifier])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
            print(np.log(resa), np.log(resb), spec)
        return relerrors
    
    def test16(c=1.0, ch=1e-3, TK=298.15): #OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['K+', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("K+", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch*nullifier, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch*nullifier])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            print(np.log(resa), np.log(resb), spec)
            relerrors.append(relerror)
        return relerrors
    
    def test17(c=1.0, ch=1e-3, TK=298.15): #OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Br-', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Br-", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch*nullifier, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch*nullifier, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            print(np.log(resa) - np.log(resb), spec)
            relerrors.append(relerror)
        return relerrors
    
    def test18(c=1.0, ch=1e-3, TK=298.15): #Not OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['CO3-2', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("CO3-2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch*nullifier, "mol")
        rkstate.set("OH-", ch, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch*nullifier, ch])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            print(np.log(resa), np.log(resb), spec)
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
        return relerrors
    
    def test19(c=1.0, ch=1e-3, TK=298.15): #OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Fe+2', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Fe+2", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch*nullifier, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch*nullifier])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
            print(np.log(resa), np.log(resb), spec)
        return relerrors
    
    def test20(c=1.0, ch=1e-3, TK=298.15): #OK
        nullifier = 1e-10
        rkdb = reaktoro.PhreeqcDatabase("phreeqc.dat")
        specs = ['Li+', 'H+', 'OH-']
        rksolution = reaktoro.AqueousPhase('H2O ' + ' '.join(specs))
        rksolution.setActivityModel(reaktoro.ActivityModelPitzerHMW())
        rksystem = reaktoro.ChemicalSystem(rkdb, rksolution)
        rkstate = reaktoro.ChemicalState(rksystem)
        rkstate.temperature(TK, "K")
        rkstate.pressure(1.0, "atm")
        rkstate.set("H2O", 1.0, "kg")
        rkstate.set("Li+", c, "mol")
        #rkstate.set("Cl-", c, "mol")
        rkstate.set("H+", ch, "mol")
        rkstate.set("OH-", ch*nullifier, "mol")
        rkprops = reaktoro.ChemicalProps(rkstate)
        solutes_pq = list(map(convert, specs))
        print(solutes_pq)
        pq_array = np.array([c, ch, ch*nullifier])
        model = pyequion2.activity.setup_pitzer(solutes_pq)
        acts = 10**model(pq_array, TK)[1:]
        relerrors = []
        for i, spec in enumerate(specs):
            resa = float(rkprops.speciesActivityCoefficient(spec))
            resb = acts[i]
            relerror = np.abs(resa - resb)/np.abs(resb)
            relerrors.append(relerror)
            print(np.log(resa), np.log(resb))
        return relerrors

#print(test6(1e0))
#print(test14(10, 1e0))
method_list = [func for func in dir(Tests) if callable(getattr(Tests, func))
               and not func.startswith("__")]
for method in method_list:
    print(method)
    print(getattr(Tests, method)())
    print('--')
# print(Tests.test16(1, 1e0))
# print(Tests.test18(1, 1e0))
# print(Tests.test19(1, 1e0))
# #print(test20(10, 1e0))
# #print(test19(10, 1e0))
# #print(test12(10, 1e0))
# #print(test13(10, 1e0))