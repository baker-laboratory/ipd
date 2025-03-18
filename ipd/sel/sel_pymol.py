import contextlib
import ipd

with contextlib.suppress(ImportError):
    import lark

    with ipd.dev.open_package_file('sel/pymol_selection_algebra.lark') as pymol_grammar:
        pymol_selection_parser = lark.Lark(pymol_grammar)  #, parser='lalr')

    def pymol(sel):
        tree = pymol_selection_parser.parse(sel)
        transformer = SelectionTransformer(None)
        result = transformer.transform(tree)
        print(result)

    @lark.v_args
    class SelectionTransformer(lark.Transformer):

        def __init__(self, atoms):
            self.atoms = atoms
            self.vars = {}

        def start(self, items):
            return items[0]

        def selection(self, items):
            return items[0]

        def maybe_assign_expr(self, items):
            return items[0]

        def assign_expr(self, items):
            return {"type": "assignment", "name": items[0], "value": items[2]}

        def or_expr(self, items):
            if len(items) == 1:
                return items[0]
            return {"type": "or_expr", "operands": items}

        def and_expr(self, items):
            if len(items) == 1:
                return items[0]
            return {"type": "and_expr", "operands": items}

        def atom_expr(self, items):
            return items[0]

        def maybe_not_expr(self, items):
            return items[0]

        def not_expr(self, items):
            return {"type": "not_expr", "operand": items[0]}

        def primary_selection(self, items):
            return items[0]

        def generic_selection(self, items):
            return {"type": "generic_selection", "value": items[0].value}

        def named_selection(self, items):
            if len(items) == 1:
                return {"type": "named_selection", "name": items[0].value}
            elif items[0].value == "%":
                return {"type": "named_selection_percent", "name": items[1].value}
            elif items[0].value == "?":
                return {"type": "named_selection_question", "name": items[1].value}

        def first_last_selection(self, items):
            return {"type": f"{items[0].value}_selection", "selection": items[1]}

        def identifier_selection(self, items):
            return items[0]

        def identifier_sel_str(self, items):
            return {"type": "identifier_sel_str", "identifier_type": items[0].value, "values": items[1]}

        def identifier_sel_int(self, items):
            return {"type": "identifier_sel_int", "identifier_type": items[0].value, "values": items[1]}

        def INT_RANGE(self, items):
            if len(items) == 1:
                return {"type": "single_int", "start": int(items[0].value)}
            return {"type": "int_range", "start": int(items[0].value), "end": int(items[2].value)}

        def RESI_LIST(self, items):
            return {"type": "resi_list", "ranges": items}

        def INT_LIST(self, items):
            return {"type": "int_list", "ranges": items}

        def STRING_LIST(self, items):
            if len(items) == 1:
                return {"type": "string_value", "value": self._clean_string(items[0].value)}
            return {
                "type": "string_list",
                "values": [self._clean_string(item.value) for item in items if item.value != "+"]
            }

        def ELEM(self, items):
            return {"type": "element", "symbol": items[0].value}

        def ELEM_LIST(self, items):
            elements = [item for item in items if item.value != "+"]
            return {"type": "element_list", "elements": [elem.value for elem in elements]}

        def id_matching_selection(self, items):
            return {"type": f"id_matching_{items[1].value}", "left": items[0], "right": items[2]}

        def entity_expansion(self, items):
            return {"type": "entity_expansion", "method": items[0].value, "selection": items[1]}

        def bond_expansion(self, items):
            return items[0]

        def bound_to_selection(self, items):
            return {"type": "bound_to", "selection": items[1]}

        def neighbor_selection(self, items):
            return {"type": "neighbor", "selection": items[1]}

        def extend_selection(self, items):
            return {"type": "extend", "selection": items[0], "distance": float(items[2].value)}

        def proximity_selection(self, items):
            return items[0]

        def within_selection(self, items):
            return {"type": "within", "selection": items[0], "distance": float(items[2].value), "target": items[4]}

        def around_selection(self, items):
            return {"type": "around", "selection": items[0], "distance": float(items[2].value)}

        def expand_selection(self, items):
            return {"type": "expand", "selection": items[0], "distance": float(items[2].value)}

        def gap_selection(self, items):
            return {"type": "gap", "selection": items[0], "distance": float(items[2].value)}

        def near_to_selection(self, items):
            return {"type": "near_to", "selection": items[0], "distance": float(items[2].value), "target": items[4]}

        def beyond_selection(self, items):
            return {"type": "beyond", "selection": items[0], "distance": float(items[2].value), "target": items[4]}

        def property_selection(self, items):
            return items[0]

        def partial_charge_selection(self, items):
            return {"type": "partial_charge", "comparison": items[1]}

        def formal_charge_selection(self, items):
            return {"type": "formal_charge", "comparison": items[1]}

        def b_factor_selection(self, items):
            return {"type": "b_factor", "comparison": items[1]}

        def occupancy_selection(self, items):
            return {"type": "occupancy", "comparison": items[1]}

        def ss_selection(self, items):
            return {"type": "secondary_structure", "value": items[1]}

        def elem_selection(self, items):
            return {"type": "element_selection", "elements": items[1]}

        def custom_prop_selection(self, items):
            return {"type": "custom_property", "property": items[1].value, "comparison": items[2]}

        def stereo_selection(self, items):
            return {"type": "stereo", "value": items[1].value}

        def COMPARISON(self, items):
            if items[0].value == "in":
                return {"type": "in_range", "range": items[1]}

            op = items[0].value
            value = items[1].value

            # Convert to the appropriate type
            try:
                value = int(value)
            except ValueError:
                value = float(value)

            return {"type": "comparison", "operator": op, "value": value}

        def COMP_OP(self, items):
            return items[0]

        def SS_TYPE(self, items):
            return {"type": "ss_type", "value": self._clean_string(items[0].value)}

        def flag_selection(self, items):
            if len(items) == 1:
                return {"type": "flag", "flag_type": items[0].value}
            else:
                # For "flag" INT case
                return {"type": "flag", "flag_type": items[0].value, "flag_number": int(items[1].value)}

        def chemical_class(self, items):
            return {"type": "chemical_class", "class_type": items[0].value}

        def coordinate_selection(self, items):
            return items[0]

        def state_selection(self, items):
            return {"type": "state", "state_num": int(items[1].value)}

        def present_selection(self, items):
            return {"type": "present"}

        def coord_comparison(self, items):
            return {"type": "coordinate_comparison", "axis": items[0].value, "comparison": items[1]}

        def atom_typing(self, items):
            return items[0]

        def text_type_selection(self, items):
            return {"type": "text_type", "value": self._clean_string(items[1].value)}

        def numeric_type_selection(self, items):
            return {"type": "numeric_type", "value": int(items[1].value)}

        def path_selection(self, items):
            return {"type": "path", "path": items[0].value}

        def _clean_string(self, s):
            """Remove quotes from strings if present."""
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
            return s
