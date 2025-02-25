"""
pymol_selection.py - A library for using PyMOL selection syntax with biotite AtomArrays

This module provides functionality to parse PyMOL selection syntax and apply it to
biotite.structure.AtomArray objects, returning boolean masks that can be used for filtering.
"""

import re
import numpy as np
import pyparsing as pp

class PyMOLSelectionParser:
    """
    Parser for PyMOL selection language to be used with biotite AtomArrays.

    This class implements PyMOL's selection language as described in the PyMOL wiki
    (https://pymolwiki.org/index.php/Selection_Algebra) and applies it to
    biotite.structure.AtomArray objects to create boolean masks.
    """

    def __init__(self):
        """Initialize the PyMOL selection parser."""
        # Define mappings between PyMOL and biotite attribute names
        self.attr_map = {
            'name': 'atom_name',
            'resn': 'res_name',
            'resi': 'res_id',
            'chain': 'chain_id',
            'element': 'element',
            'b': 'b_factor',
            'q': 'occupancy',
            'x': lambda arr: arr.coord[:, 0],
            'y': lambda arr: arr.coord[:, 1],
            'z': lambda arr: arr.coord[:, 2],
            'alt': 'altloc_id',
            'index': lambda arr: np.arange(len(arr)),
            'id': lambda arr: np.arange(1,
                                        len(arr) + 1)  # 1-based like PyMOL
        }

        # Define selection macros
        self.selection_macros = {
            'all': self._select_all,
            'none': self._select_none,
            'backbone': self._select_backbone,
            'sidechain': self._select_sidechain,
            'protein': self._select_protein,
            'nucleic': self._select_nucleic,
            'water': self._select_water,
            'hydrogen': self._select_hydrogen,
            'carbon': self._select_carbon,
            'nitrogen': self._select_nitrogen,
            'oxygen': self._select_oxygen,
            'sulfur': self._select_sulfur,
            'polar': self._select_polar,
            'nonpolar': self._select_nonpolar,
            'charged': self._select_charged,
            'aromatic': self._select_aromatic,
            'hydrophobic': self._select_hydrophobic
        }

        # Initialize the parser
        self._setup_parser()

    def _setup_parser(self):
        """Set up the pyparsing grammar for PyMOL selection language."""
        # Define basic elements
        identifier = pp.Word(pp.alphas + '_', pp.alphanums + '_')
        integer = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
        real = pp.Combine(pp.Optional(pp.Char('+-')) + pp.Word(pp.nums) +
                          pp.Optional(pp.Char('.') + pp.Word(pp.nums))).setParseAction(lambda t: float(t[0]))
        number = real | integer

        # Define string literals
        quoted_string = (pp.QuotedString('"', escChar='\\') | pp.QuotedString("'", escChar='\\'))

        # Fix the wildcard_string definition to better handle patterns with '-'
        wildcard_string = pp.Regex(r'[a-zA-Z0-9_*]+(?:-[a-zA-Z0-9_*]+)*')

        # Define selection expressions
        selection_expr = pp.Forward()

        # Selection macros
        selection_macro = pp.oneOf(' '.join(self.selection_macros.keys()))

        # Atom properties with comparisons
        atom_property = identifier
        eq_operator = pp.Literal('=')
        ne_operator = pp.Literal('!=')
        gt_operator = pp.Literal('>')
        lt_operator = pp.Literal('<')
        ge_operator = pp.Literal('>=')
        le_operator = pp.Literal('<=')
        comparison_operator = (eq_operator | ne_operator | gt_operator | lt_operator | ge_operator | le_operator)

        # Value can be a number, quoted string, or wildcard string
        value = quoted_string | wildcard_string | number

        # Multiple values in comma-separated list
        value_list = pp.delimitedList(value, delim=',')

        # Property comparison
        property_comparison = (atom_property + comparison_operator + (value_list | value))

        # Range selection (e.g., resi 10-20)
        range_selector = (atom_property + integer + pp.Suppress('-') + integer)

        # Parenthesized expression
        parenthesized = pp.Forward()
        parenthesized << (pp.Suppress('(') + selection_expr + pp.Suppress(')'))

        # Boolean operators
        and_op = pp.Keyword('and') | pp.Literal('&')
        or_op = pp.Keyword('or') | pp.Literal('|')
        not_op = pp.Keyword('not') | pp.Literal('!')

        # Proximity selections
        within_selector = (pp.Keyword('within') + number + pp.Keyword('of') + selection_expr)

        # Simple terms
        simple_term = (property_comparison | range_selector | selection_macro | within_selector)

        # Negation
        negated_term = not_op + (simple_term | parenthesized)

        # Term
        term = negated_term | simple_term | parenthesized

        # Expression with precedence
        and_expr = pp.Forward()
        or_expr = pp.Forward()
        and_expr << (term + pp.ZeroOrMore(and_op + term))
        or_expr << (and_expr + pp.ZeroOrMore(or_op + and_expr))

        # Set top-level expression
        selection_expr << or_expr

        # Store the parser
        self.parser = selection_expr

    def __call__(self, atoms, sel):
        return self.select(atoms, sel)

    def select(self, atoms, sel):
        """
        Parse a PyMOL selection string and apply it to an AtomArray.

        Parameters
        ----------
        atoms : biotite.structure.AtomArray
            The atom array to select from.
        sel : str
            PyMOL selection syntax.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask representing the selection.
        """
        # Handle empty or whitespace-only selection
        sel = sel.strip()
        if not sel:
            return np.ones(len(atoms), dtype=bool)

        print(sel)

        try:
            # Parse the selection string
            parsed = self.parser.parseString(sel, parseAll=True)
            # Evaluate the parsed expression
            return self._evaluate_parsed_expression(atoms, parsed)
        except pp.ParseException as e:
            raise ValueError(f"Failed to parse selection: {str(e)}")

    # def _evaluate_parsed_expression(self, atoms, parsed_expr):
    #     """Evaluate a parsed selection expression."""
    #     # This is a simplified evaluation - a full implementation would
    #     # handle the complete parsing tree

    #     # For demonstration purposes, we'll handle a few common cases
    #     if len(parsed_expr) == 1:
    #         # Single term (like a macro or property comparison)
    #         term = parsed_expr[0]
    #         if isinstance(term, str) and term in self.selection_macros:
    #             return self.selection_macros[term](atoms)
    #         elif isinstance(term, pp.ParseResults) and len(term) >= 3:
    #             # Property comparison
    #             prop, op, value = term[0], term[1], term[2]
    #             return self._evaluate_comparison(atoms, prop, op, value)
    #     elif len(parsed_expr) == 2 and parsed_expr[0] in ('not', '!'):
    #         # Negation
    #         result = self._evaluate_parsed_expression(atoms, [parsed_expr[1]])
    #         return ~result
    #     elif len(parsed_expr) >= 3 and parsed_expr[1] in ('and', '&', 'or', '|'):
    #         # Boolean operation
    #         left = self._evaluate_parsed_expression(atoms, [parsed_expr[0]])
    #         op = parsed_expr[1]
    #         right = self._evaluate_parsed_expression(atoms, [parsed_expr[2]])

    #         if op in ('and', '&'):
    #             return left & right
    #         elif op in ('or', '|'):
    #             return left | right

    #     # Fallback to simple tokenization and evaluation
    #     return self._parse_selection_simple(atoms, " ".join(str(p) for p in parsed_expr))

    def _evaluate_parsed_expression(self, atom_array, parsed_expr):
        """Evaluate a parsed selection expression."""
        if len(parsed_expr) == 0:
            return np.ones(len(atom_array), dtype=bool)

        if len(parsed_expr) == 1:
            # Single term (like a macro or property comparison)
            term = parsed_expr[0]
            if isinstance(term, str) and term in self.selection_macros:
                return self.selection_macros[term](atom_array)
            elif isinstance(term, pp.ParseResults):
                # Property comparison
                if len(term) >= 3 and term[1] in ('=', '!=', '>', '<', '>=', '<='):
                    # Standard comparison
                    prop, op, value = term[0], term[1], term[2]
                    return self._evaluate_comparison(atom_array, prop, op, value)
                elif len(term) == 4 and term[1] == 'range':
                    # Range selection (e.g., "resi 10-20")
                    prop, start, end = term[0], term[2], term[3]
                    biotite_attr = self._get_biotite_attribute(atom_array, prop)
                    return (biotite_attr >= start) & (biotite_attr <= end)
        elif len(parsed_expr) == 2 and parsed_expr[0] in ('not', '!'):
            # Negation
            result = self._evaluate_parsed_expression(atom_array, [parsed_expr[1]])
            return ~result
        elif len(parsed_expr) >= 3 and parsed_expr[1] in ('and', '&', 'or', '|'):
            # Boolean operation
            left = self._evaluate_parsed_expression(atom_array, [parsed_expr[0]])
            op = parsed_expr[1]
            right = self._evaluate_parsed_expression(atom_array, [parsed_expr[2]])
            if op in ('and', '&'):
                return left & right
            elif op in ('or', '|'):
                return left | right

        # Fallback to simple tokenization and evaluation
        return self._parse_selection_simple(atom_array, " ".join(str(p) for p in parsed_expr))

    def _parse_selection_simple(self, atoms, sel):
        """Simple fallback parser for handling basic selections."""
        # This is a simplified version that handles basic selections
        # when the full parser doesn't handle the expression

        # Check for selection macros
        if sel in self.selection_macros:
            return self.selection_macros[sel](atoms)

        # Handle basic AND/OR expressions
        if " and " in sel or " & " in sel:
            delimiter = " and " if " and " in sel else " & "
            parts = sel.split(delimiter, 1)
            left = self._parse_selection_simple(atoms, parts[0])
            right = self._parse_selection_simple(atoms, parts[1])
            return left & right

        if " or " in sel or " | " in sel:
            delimiter = " or " if " or " in sel else " | "
            parts = sel.split(delimiter, 1)
            left = self._parse_selection_simple(atoms, parts[0])
            right = self._parse_selection_simple(atoms, parts[1])
            return left | right

        # Handle NOT expressions
        if sel.startswith("not ") or sel.startswith("! "):
            expr = sel[4:] if sel.startswith("not ") else sel[2:]
            result = self._parse_selection_simple(atoms, expr)
            return ~result

        # Handle basic comparisons
        for op in ['>=', '<=', '!=', '=', '>', '<']:
            if op in sel:
                parts = sel.split(op, 1)
                attribute = parts[0].strip()
                value = parts[1].strip()
                return self._evaluate_comparison(atoms, attribute, op, value)

        # Handle range selections (e.g., "resi 10-20")
        if ' ' in sel and '-' in sel:
            parts = sel.split(' ', 1)
            attribute = parts[0].strip()
            value_range = parts[1].strip()
            if '-' in value_range:
                start, end = value_range.split('-', 1)
                try:
                    start_val = int(start.strip())
                    end_val = int(end.strip())
                    biotite_attr = self._get_biotite_attribute(atoms, attribute)
                    return (biotite_attr >= start_val) & (biotite_attr <= end_val)
                except ValueError:
                    pass

        # Default to all atoms if we can't parse
        return np.ones(len(atoms), dtype=bool)

    def _evaluate_comparison(self, atoms, attribute, operator, value):
        """Evaluate a comparison expression."""
        # Get the attribute values
        biotite_attr = self._get_biotite_attribute(atoms, attribute)

        # Handle comma-separated lists
        if isinstance(value, str) and ',' in value:
            values = [v.strip() for v in value.split(',')]
            masks = [self._compare_single_value(biotite_attr, operator, v) for v in values]
            return np.logical_or.reduce(masks)
        else:
            return self._compare_single_value(biotite_attr, operator, value)

    def _compare_single_value(self, attribute_values, operator, value):
        """Compare attribute values with a single value using the specified operator."""
        # Remove quotes if present
        if isinstance(value, str):
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

        # Handle wildcards in string comparisons
        if isinstance(value, str) and '*' in value and isinstance(attribute_values[0], str):
            pattern = value.replace('*', '.*')
            regex = re.compile(f"^{pattern}$")

            # Apply regex to each value
            mask = np.zeros(len(attribute_values), dtype=bool)
            for i, val in enumerate(attribute_values):
                if regex.match(val):
                    mask[i] = True
            return mask

        # Convert value to numeric if comparing with numbers
        if operator in ['>', '<', '>=', '<='] and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Cannot convert {value} to numeric for comparison")

        # Apply the comparison
        if operator == '=' or operator == '==':
            return attribute_values == value
        elif operator == '!=':
            return attribute_values != value
        elif operator == '>':
            return attribute_values > value
        elif operator == '<':
            return attribute_values < value
        elif operator == '>=':
            return attribute_values >= value
        elif operator == '<=':
            return attribute_values <= value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _get_biotite_attribute(self, atoms, attribute):
        """Get values of an attribute from the atom array."""
        if attribute in self.attr_map:
            attr_accessor = self.attr_map[attribute]
            if callable(attr_accessor):
                return attr_accessor(atoms)
            else:
                return getattr(atoms, attr_accessor)
        else:
            # Try direct access
            try:
                return getattr(atoms, attribute)
            except AttributeError:
                raise ValueError(f"Unknown attribute: {attribute}")

    # Selection macro methods
    def _select_all(self, atoms):
        """Select all atoms."""
        return np.ones(len(atoms), dtype=bool)

    def _select_none(self, atoms):
        """Select no atoms."""
        return np.zeros(len(atoms), dtype=bool)

    def _select_backbone(self, atoms):
        """Select backbone atoms (N, CA, C, O)."""
        backbone_names = ["N", "CA", "C", "O"]
        mask = np.zeros(len(atoms), dtype=bool)
        for name in backbone_names:
            mask |= atoms.atom_name == name
        return mask & self._select_protein(atoms)

    def _select_sidechain(self, atoms):
        """Select sidechain atoms (not backbone)."""
        backbone_mask = self._select_backbone(atoms)
        return ~backbone_mask & self._select_protein(atoms)

    def _select_protein(self, atoms):
        """Select protein atoms."""
        # Common amino acid three-letter codes
        aa_codes = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "MSE",
            "SEC",
            "PYL",
            "UNK"  # Some non-standard amino acids
        }
        mask = np.zeros(len(atoms), dtype=bool)
        for aa in aa_codes:
            mask |= atoms.res_name == aa
        return mask

    def _select_nucleic(self, atoms):
        """Select nucleic acid atoms."""
        # Nucleic acid residue codes
        na_codes = {
            "A",
            "C",
            "G",
            "T",
            "U",
            "DA",
            "DC",
            "DG",
            "DT",
            "ADE",
            "CYT",
            "GUA",
            "THY",
            "URA",
            "5MC",
            "PSU",
            "5MU",
            "H2U"  # Include modified bases
        }
        mask = np.zeros(len(atoms), dtype=bool)
        for na in na_codes:
            mask |= atoms.res_name == na
        return mask

    def _select_water(self, atoms):
        """Select water molecules."""
        water_names = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL"}
        mask = np.zeros(len(atoms), dtype=bool)
        for name in water_names:
            mask |= atoms.res_name == name
        return mask

    def _select_hydrogen(self, atoms):
        """Select hydrogen atoms."""
        return atoms.element == "H"

    def _select_carbon(self, atoms):
        """Select carbon atoms."""
        return atoms.element == "C"

    def _select_nitrogen(self, atoms):
        """Select nitrogen atoms."""
        return atoms.element == "N"

    def _select_oxygen(self, atoms):
        """Select oxygen atoms."""
        return atoms.element == "O"

    def _select_sulfur(self, atoms):
        """Select sulfur atoms."""
        return atoms.element == "S"

    def _select_polar(self, atoms):
        """Select polar atoms (N, O, S atoms and hydrogens bonded to them)."""
        # This is a simplified implementation
        polar_elements = {"N", "O", "S"}
        mask = np.zeros(len(atoms), dtype=bool)
        for elem in polar_elements:
            mask |= atoms.element == elem
        return mask

    def _select_nonpolar(self, atoms):
        """Select nonpolar atoms."""
        return ~self._select_polar(atoms)

    def _select_charged(self, atoms):
        """Select charged atoms."""
        # Simplified implementation - in a real implementation,
        # we would need to determine charges based on residue and atom type
        charged_atoms = {("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"),
                         ("GLU", "OE2")}
        mask = np.zeros(len(atoms), dtype=bool)
        for i in range(len(atoms)):
            if (atoms.res_name[i], atoms.atom_name[i]) in charged_atoms:
                mask[i] = True
        return mask

    def _select_aromatic(self, atoms):
        """Select aromatic atoms (atoms in aromatic rings)."""
        # Simplified implementation - focus on residues with aromatic rings
        aromatic_residues = {"PHE", "TYR", "TRP", "HIS"}
        aromatic_atom_patterns = {
            "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]
        }

        mask = np.zeros(len(atoms), dtype=bool)
        for i in range(len(atoms)):
            res_name = atoms.res_name[i]
            atom_name = atoms.atom_name[i]

            if res_name in aromatic_residues:
                if atom_name in aromatic_atom_patterns.get(res_name, []):
                    mask[i] = True

        return mask

    def _select_hydrophobic(self, atoms):
        """Select hydrophobic atoms (typically carbons and attached hydrogens in nonpolar residues)."""
        # Simplified implementation - nonpolar residues
        hydrophobic_residues = {"ALA", "LEU", "ILE", "VAL", "PHE", "PRO", "MET"}
        mask = np.zeros(len(atoms), dtype=bool)

        for res in hydrophobic_residues:
            res_mask = atoms.res_name == res
            element_mask = (atoms.element == "C") | (atoms.element == "H")
            mask |= res_mask & element_mask

        return mask

# Convenience function for simple usage
def select(atoms, sel):
    """
    Apply a PyMOL selection string to a biotite AtomArray.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        The atom array to select from.
    sel : str
        PyMOL selection syntax.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask representing the selection.
    """
    parser = PyMOLSelectionParser()
    return parser.select(atoms, sel)
