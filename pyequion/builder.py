# -*- coding: utf-8 -*-
import re
import copy

import numpy as np
import ordered_set
import commentjson

from . import datamods


ELEMENT_SPECIES_MAP = {
    "C": "HCO3-",
    "Ca": "Ca++",
    "Cl": "Cl-",
    "Na": "Na+",
    "S": "SO4--",
    "Ba": "Ba++",
    "Mg": "Mg++",
    'Fe': 'Fe++',
    'K': 'K+',
    'Sr': 'Sr++',
    'N': 'NH3',
    'Cd': 'Cd++',
    'Li': 'Li+',
    'Cu': 'Cu+',
    'Al': 'Al+++',
    'Br': 'Br-',
    'F': 'F-',
    'Mn': 'Mn++',
    'P': 'PO4---',
    'Pb': 'Pb++',
    'Zn': 'Zn++'
}
DEFAULT_DB_FILES = {
    "solutions": datamods.reactions_solutions,
    "phases": datamods.reactions_solids,
    "irreversible": datamods.reactions_irreversible,
    "species": datamods.species,
    "gases": datamods.reactions_gases
}
ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
            'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
            'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
            'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
            'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
            'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
            'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts',
            'Og']
RX_CASE = r"[A-Z][^A-Z]*"
RX_NO_SIGNAL = r"[+-]"
RX_PRNTHS = r"(\(\w+\)\d?)"
RX_DIGIT = "[A-Z][^A-Z]*"


def elements_to_species(element_set):
    """
    Get species that are placeholders for the elements, and adds H2O
    """
    species = {ELEMENT_SPECIES_MAP[el] for el in element_set
               if el not in ['O', 'H']}
    species.add('H2O')
    return species


def species_to_elements(species_set):
    """
    Get elements that are in the set of species
    """
    return _get_element_set_from_comp_list(list(species_set))


def get_species_reaction_from_initial_species(initial_species,
                                              possible_reactions=None,
                                              possible_solid_reactions=None,
                                              possible_gas_reactions=None,
                                              database_files=None):
    if database_files is None:
        database_files = DEFAULT_DB_FILES
    if not possible_reactions:
        possible_reactions = get_all_possible_reactions(database_files)
    if not possible_solid_reactions:
        possible_solid_reactions = get_all_possible_solid_reactions(database_files)
    if not possible_gas_reactions:
        possible_gas_reactions = get_all_possible_gas_reactions(database_files)
    species, reactions = _get_species_reactions_from_compounds(
        initial_species, possible_reactions)
    species = list(species)
    species = set_h2o_as_first_specie(species)
    _, solid_reactions = _get_species_reactions_from_compounds(
        set(species), possible_solid_reactions
    )
    _, gas_reactions = _get_species_reactions_from_compounds(
            set(species), possible_gas_reactions)
    species, reactions, solid_reactions, gas_reactions = \
        _sort_species_and_reactions(species, reactions, solid_reactions, gas_reactions)
    return species, reactions, solid_reactions, gas_reactions


def get_all_possible_reactions(database_files=DEFAULT_DB_FILES):
    aqueous_reactions = load_from_db(database_files["solutions"])
    irreversible_reactions = load_from_db(database_files["irreversible"])
    possible_reactions = aqueous_reactions + irreversible_reactions
    return possible_reactions


def get_all_possible_solid_reactions(database_files=DEFAULT_DB_FILES):
    possible_solid_reactions = load_from_db(database_files["phases"])
    return possible_solid_reactions


def get_all_possible_gas_reactions(database_files=DEFAULT_DB_FILES):
    possible_gas_reactions = load_from_db(database_files["gases"])
    return possible_gas_reactions


def get_log_equilibrium_constants(reactions, TK, PATM):
    return np.array([_get_logk(reaction, TK, PATM) for reaction in reactions])


def make_formula_matrix(species, elements):
    elements = elements + ['e']
    formula_matrix = np.array([[stoich_number(specie, element)
                                for specie in species]
                               for element in elements])
    return formula_matrix


def make_solid_formula_matrix(solid_reactions, elements):
    solid_formulas = [_get_solid_formula(solid_reaction)
                      for solid_reaction in solid_reactions]
    elements = elements + ['e']
    solid_formula_matrix = np.array([[stoich_number(solid_formula, element)
                                      for solid_formula in solid_formulas]
                                     for element in elements])
    return solid_formula_matrix


def make_gas_formula_matrix(gas_reactions, elements):
    gas_formulas = [_get_gas_formula(gas_reaction)
                      for gas_reaction in gas_reactions]
    elements = elements + ['e']
    gas_formula_matrix = np.array([[stoich_number(gas_formula, element)
                                      for gas_formula in gas_formulas]
                                     for element in elements])
    return gas_formula_matrix


def make_stoich_matrix(species, reactions):
    return np.array([[reaction.get(specie, 0.0) for specie in species]
                     for reaction in reactions])


def set_h2o_as_first_specie(species):
    try:
        species.pop(species.index('H2O'))
    except:
        pass
    species = ['H2O'] + species
    return species


def set_h_and_o_as_first_elements(elements):
    try:
        elements.pop(elements.index('H'))
    except:
        pass
    try:
        elements.pop(elements.index('O'))
    except:
        pass
    elements = ['H', 'O'] + elements
    return elements


def get_most_stable_phases(solid_reactions, TK, PATM):
    stable_phases_group = dict()
    stable_lowest_ksp = dict()
    log_ksps = get_log_equilibrium_constants(solid_reactions, TK, PATM)
    for i, solid_reaction in enumerate(solid_reactions):
        phase_name = solid_reaction['phase_name']
        solid_formula = _get_solid_formula(
            solid_reaction, drop_phase_name=False)
        lowest_ksp = stable_lowest_ksp.get(solid_formula, np.inf)
        current_ksp = log_ksps[i]
        if current_ksp < lowest_ksp:
            stable_phases_group[solid_formula] = phase_name
            stable_lowest_ksp[solid_formula] = current_ksp
    stable_phases = list(stable_phases_group.values())
    return stable_phases


def get_elements_and_their_coefs(list_of_species):
    # Removing signals
    tags_no_signals = [
        re.sub(RX_NO_SIGNAL, "", tag) for tag in list_of_species
    ]
    # Clean-up phase name if it has it
    tags_no_signals = [tag.split("__")[0] for tag in tags_no_signals]

    elements_with_coefs = [
        _separate_elements_coefs(item) for item in tags_no_signals
    ]

    return elements_with_coefs


def stoich_number(specie, element):
    """
    Get stoichometric coeficient of element in specie
    """
    if element == 'e':  # Charge number
        return charge_number(specie)
    else:
        elements_and_coefs = dict(get_elements_and_their_coefs([specie])[0])
        coef = elements_and_coefs.get(element, 0)
        return coef


def charge_number(specie):
    """
    Get charge number of specie
    """
    return 1*specie.count('+') - 1*specie.count('-')


def load_from_db(fname):
    if not isinstance(fname, str):
        return copy.deepcopy(fname)
    #Open a json file as dict
    with open(fname, "r") as json_file:
        db = commentjson.load(json_file)
    return db


def _get_logk(reaction, TK, PATM):
    log_K_coefs = reaction.get('log_K_coefs', '')
    if type(log_K_coefs) != str:
        log_K_patm = _calculate_logk_1(log_K_coefs, TK)
    else:
        log_K25 = reaction.get('log_K25', '')
        if type(log_K25) != str:
            deltah = reaction.get('deltah', 0.0)
            if type(deltah) == str:
                deltah = 0.0
            log_K_patm = _calculate_logk_2(log_K25, deltah, TK)
        else:
            log_K_patm = 0.0  # Can't do nothing beyond this
    log_K = log_K_patm + _adjust_pressure(reaction, TK, PATM)
    return log_K


def _calculate_logk_1(log_K_coefs, TK):
    log_K_coefs = np.hstack([log_K_coefs, np.zeros(6-len(log_K_coefs))])
    temperature_array = np.array([1, TK, 1/TK, np.log10(TK), 1/(TK**2), TK**2])
    logK = np.sum(log_K_coefs*temperature_array)
    return logK


def _calculate_logk_2(log_K25, deltah, TK):
    T0 = 298.15
    R = 8.314
    logK = log_K25 - deltah/(2.303 * R)*(1/TK - 1/T0)
    return logK


def _adjust_pressure(reaction, TK, PATM):
    vm_coefs = reaction.get('Vm', '')
    if vm_coefs == '':
        return 0.0 #No adjustment
    else:
        if TK <= 229: #Out of validity
            return 0.0            
        R = 8.314
        Pb = 1.01325*PATM
        vm_coefs = vm_coefs + [0.0]*(10-len(vm_coefs)) #Padding
        a1, a2, a3, a4, W, ao, i1, i2, i3, i4 = vm_coefs
        phi_inf = 41.84*(
            a1*0.1 + 
            a2*100/(2600+Pb) + 
            a3/(TK - 228) + 
            a4*1e4/((Pb + 2600)*(TK - 228)))
        #Due to other terms depending on ionic strength
        #(which depends on the composition itself)
        #They won't be calculated
        vm = phi_inf #+ Debye-Huckel terms + Born volume therms
        #Volumes can be negative because it is actually a volume change
        adjustment = -vm/(R*TK)*(Pb - 1.01325)
        return adjustment

def _get_reactions_species(reac):
    return [k for k in reac.keys()
            if _check_validity_specie_tag_in_reaction_dict(k)]


def _get_species_reactions_from_compounds(compounds, possible_reactions):
    species = ordered_set.OrderedSet([c for c in compounds])
    reactions = []
    for c in compounds:
        _walk_in_species_reactions(c, species, reactions, possible_reactions)
    species = list(species)
    return species, reactions


def _get_element_set_from_comp_list(comp_list):
    """
    Get elements that are in the set of species
    """
    list_elements_in_tags = get_elements_and_their_coefs(comp_list)
    aux_ele_as_list = [[sub[0] for sub in item]
                       for item in list_elements_in_tags]
    aux_ele_flat = [sub for item in aux_ele_as_list for sub in item]
    ele_set = set(aux_ele_flat)
    return ele_set


def _separate_elements_coefs(tag):
    separated_parens = re.split(RX_PRNTHS, tag)
    separated_parens = [val for val in separated_parens if val != ""]
    elements_with_coefs = []
    for v in separated_parens:
        if "(s)" in v or "(g)" in v:
            continue
        if "(" in v and ")" in v:
            intr_prh = v[1:-2]
            d = int(v[-1])
            case_coefs = _get_tag_el_coef(intr_prh)
            for el_coef in case_coefs:
                el_coef[1] *= d
        elif ":" in v:
            d_first = int(v[1]) if v[1].isdigit() else 1
            case_coefs = _get_tag_el_coef(v[1:])
            for el_coef in case_coefs:
                el_coef[1] *= d_first
        else:
            # by_case = re.findall(RX_CASE, v)
            case_coefs = _get_tag_el_coef(v)
        elements_with_coefs += case_coefs
    return elements_with_coefs


def _get_tag_el_coef(tag):
    by_case = re.findall(RX_CASE, tag)
    case_coefs = [_separate_letter_digit(e) for e in by_case]
    return case_coefs


def _separate_letter_digit(el_tag):
    match_dig = re.match(r"([aA-zZ]+)([0-9]+)", el_tag)
    if match_dig:
        el, dig = match_dig.groups()
    else:
        el, dig = el_tag, 1
    return [el, int(dig)]


def _get_tags_of_prods_reactions(r: dict):
    """Get Reaction Species tags for products and reactions"""
    prods = [
        k
        for k, v in r.items()
        if _check_validity_specie_tag_in_reaction_dict(k)
        if v > 0
    ]
    reacs = [
        k
        for k, v in r.items()
        if _check_validity_specie_tag_in_reaction_dict(k)
        if v < 0
    ]
    return prods, reacs


def _check_validity_specie_tag_in_reaction_dict(k):
    """Validate key in database reaction entry"""
    return not (
        k == "type"
        or k == "id_db"
        or k == "log_K25"
        or k == "log_K_coefs"
        or k == "deltah"
        or k == "phase_name"
        or k == "Omega"
        or k == "T_c"
        or k == "P_c"
        or k == "Vm"
    )


def _walk_in_species_reactions(c, species, reactions, reactionsList):
    for r in reactionsList:
        if r["type"] == "electronic":
            continue  # skipping electronic in this evaluation
        if r in reactions:
            continue
        if c not in r:
            continue
        prods, reacs = _get_tags_of_prods_reactions(r)
        # THIS WAY: SOLID ALLWAYS (-1 ; NEGATIVE)
        if (
            c in prods
            and _are_others_in_side_of_reaction_known(prods, species)
            and r["type"] != "irrev"
        ):  # IRREV ALWAYS SHOULD BE react -> prod
            reactions += [r]
            for in_element in reacs:
                if "phase_name" in r and "(s)" in in_element:
                    tag_add = in_element + "__" + r["phase_name"]
                    r[tag_add] = r.pop(in_element)  # update reaction tag!
                else:
                    tag_add = in_element
                species.add(tag_add)
                _walk_in_species_reactions(
                    in_element, species, reactions, reactionsList
                )
        if c in reacs and _are_others_in_side_of_reaction_known(reacs, species):
            reactions += [r]
            for in_element in prods:
                species.add(in_element)
                _walk_in_species_reactions(
                    in_element, species, reactions, reactionsList
                )


def _are_others_in_side_of_reaction_known(elements, species):
    tester = [e for e in elements if e in species]
    return len(tester) == len(elements)


def _get_solid_formula(solid_reaction, drop_phase_name=True):
    phase_name = solid_reaction['phase_name']
    solid_formula = None
    for key in solid_reaction.keys():
        if key[-len(phase_name):] == phase_name:
            solid_formula = key[:key.index('_')]
            if '(' in solid_formula and drop_phase_name:  # Will remove (g) or (s)
                solid_formula = solid_formula[:solid_formula.index('(')]
            break
    return solid_formula


def _get_gas_formula(gas_reaction, drop_phase_name=True):
    phase_name = gas_reaction['phase_name']
    gas_formula = None
    for key in gas_reaction.keys():
        if key == phase_name:
            gas_formula = key
            if '(' in gas_formula and drop_phase_name:  # Will remove (g) or (s)
                gas_formula = gas_formula[:gas_formula.index('(')]
            break
    return gas_formula


def _sort_species_and_reactions(species, reactions, solid_reactions, gas_reactions):
    dummy_reaction_sort_ = lambda reaction: ' '.join(sorted(reaction.keys()))
    dummy_phase_reaction_sort_ = lambda reaction: reaction['phase_name']
    reactions = sorted(reactions, key=dummy_reaction_sort_)
    solid_reactions = sorted(solid_reactions, key=dummy_phase_reaction_sort_)
    gas_reactions = sorted(gas_reactions, key=dummy_phase_reaction_sort_)
    species = [species[0]] + sorted(species[1:]) #Ensure H2O first
    return species, reactions, solid_reactions, gas_reactions
    
    
    
    
    