import pytest
import numpy as np

try:
    from biotite.structure import AtomArray
    import biotite.structure.io as strucio
except ModuleNotFoundError:
    pytest.skip(allow_module_level=True)

import ipd

parser = ipd.sel.PyMOLSelectionParser()
small_protein = ipd.tests.fixtures.small_atoms()
mixed_structure = ipd.tests.fixtures.mixed_atoms()

def main():
    # ipd.tests.maintest(namesace=globals())
    test_PyMOLSelectionParser()
    test_select_all()
    test_empty_selection()
    test_none_selection()
    test_atom_name_selection()
    test_atom_name_wildcard()
    test_residue_name_selection()
    test_chain_selection()
    test_element_selection()
    test_and_operator()
    test_or_operator()
    test_not_operator()
    test_backbone_selection()
    test_protein_selection()
    test_nucleic_selection()
    test_water_selection()
    test_not_equal_operator()
    test_single_atom_structure()
    test_empty_structure()
    test_invalid_selection_syntax()
    test_unknown_attribute()
    test_mismatched_parentheses()
    test_invalid_numeric_comparison()
    test_complex_selection_with_not()

    # test_residue_id_selection()
    # test_b_factor_selection()
    # test_altloc_selection()
    # test_complex_boolean_expression()
    # test_parentheses_precedence()
    # test_sidechain_selection()
    # test_greater_than_operator()
    # test_less_than_operator()
    # test_greater_equal_operator()
    # test_less_equal_operator()
    # test_comma_separated_values()
    # test_residue_range()
    # test_combine_macros_and_properties()
    # test_complex_selection_chain_resdue_atom()

def test_PyMOLSelectionParser():

    # Load a structure
    atom_array = strucio.load_structure(ipd.dev.package_testdata_path('pdb/px017.pdb'))

    # Apply selections
    ca_mask = parser(atom_array, "name=CA")
    chain_a_mask = parser(atom_array, "chain=A")
    active_site_mask = parser(atom_array, "resi=100-110 and chain=A")

    # Get the selected atoms
    ca_atoms = atom_array[ca_mask]
    chain_a_atoms = atom_array[chain_a_mask]
    active_site = atom_array[active_site_mask]

# Basic selection tests
def test_select_all():
    """Test 'all' selection."""
    mask = parser.select(small_protein, "all")
    assert mask.shape == (20, )
    assert np.all(mask)

def test_empty_selection():
    """Test empty selection returns all atoms."""
    mask = parser.select(small_protein, "")
    assert mask.shape == (20, )
    assert np.all(mask)

def test_none_selection():
    """Test 'none' selection."""
    mask = parser.select(small_protein, "none")
    assert mask.shape == (20, )
    assert not np.any(mask)

# Atom property selections
def test_atom_name_selection():
    """Test atom name selection."""
    mask = parser.select(small_protein, "name=CA")
    assert mask.shape == (20, )
    assert np.sum(mask) == 3  # 3 alpha carbons

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.atom_name == "CA")

def test_atom_name_wildcard():
    """Test atom name selection with wildcards."""
    mask = parser.select(small_protein, "name=C*")
    assert mask.shape == (20, )
    assert np.sum(mask) > 0

    # All selected atoms should start with C
    selected_atoms = small_protein[mask]
    for name in selected_atoms.atom_name:
        assert name.startswith("C")

def test_residue_name_selection():
    """Test residue name selection."""
    mask = parser.select(small_protein, "resn=ALA")
    assert mask.shape == (20, )
    assert np.sum(mask) == 5  # 5 atoms in ALA residue

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_name == "ALA")

@pytest.mark.xfail
def test_residue_id_selection():
    """Test residue ID selection."""
    mask = parser.select(small_protein, "resi=2")
    print(mask)
    assert mask.shape == (20, )
    assert np.sum(mask) == 8  # 8 atoms in residue 2 (PHE)

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_id == 2)

def test_chain_selection():
    """Test chain selection."""
    mask = parser.select(small_protein, "chain=A")
    assert mask.shape == (20, )
    assert np.sum(mask) == 13  # 13 atoms in chain A

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.chain_id == "A")

def test_element_selection():
    """Test element selection."""
    mask = parser.select(small_protein, "element=N")
    assert mask.shape == (20, )
    assert np.sum(mask) == 3  # 3 nitrogen atoms

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.element == "N")

@pytest.mark.xfail
def test_b_factor_selection():
    """Test b-factor selection."""
    mask = parser.select(small_protein, "b>50")
    assert mask.shape == (20, )

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.b_factor > 50)

@pytest.mark.xfail
def test_altloc_selection():
    """Test alternate location selection."""
    mask = parser.select(small_protein, "alt=A")
    assert mask.shape == (20, )
    assert np.sum(mask) == 1  # Only one atom has altloc A

    # Check the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.altloc_id == "A")

# Boolean operator tests
def test_and_operator():
    """Test AND operator."""
    # Using keyword
    mask1 = parser.select(small_protein, "name=CA and chain=A")
    assert mask1.shape == (20, )
    assert np.sum(mask1) == 2  # 2 CA atoms in chain A

    # Using symbol
    mask2 = parser.select(small_protein, "name=CA & chain=A")
    assert np.array_equal(mask1, mask2)

def test_or_operator():
    """Test OR operator."""
    # Using keyword
    mask1 = parser.select(small_protein, "name=N or name=O")
    assert mask1.shape == (20, )
    assert np.sum(mask1) == 6  # 3 N atoms and 3 O atoms

    # Using symbol
    mask2 = parser.select(small_protein, "name=N | name=O")
    assert np.array_equal(mask1, mask2)

def test_not_operator():
    """Test NOT operator."""
    # Using keyword
    mask1 = parser.select(small_protein, "not chain=A")
    assert mask1.shape == (20, )
    assert np.sum(mask1) == 7  # All atoms not in chain A

    # Using symbol
    mask2 = parser.select(small_protein, "!chain=A")
    assert np.array_equal(mask1, mask2)

@pytest.mark.xfail
def test_complex_boolean_expression():
    """Test complex boolean expression."""
    mask = parser.select(small_protein, "name=CA and (chain=A or resi=3)")
    assert mask.shape == (20, )
    assert np.sum(mask) == 3  # All CA atoms in either chain A or residue 3

@pytest.mark.xfail
def test_parentheses_precedence():
    """Test parentheses for operator precedence."""
    # Different precedence should yield different results
    mask1 = parser.select(small_protein, "name=CA and chain=A or chain=B")
    mask2 = parser.select(small_protein, "name=CA and (chain=A or chain=B)")
    mask3 = parser.select(small_protein, "(name=CA and chain=A) or chain=B")

    assert not np.array_equal(mask1, mask2) or not np.array_equal(mask1, mask3)

# Selection macro tests
def test_backbone_selection():
    """Test 'backbone' selection."""
    mask = parser.select(small_protein, "backbone")
    assert mask.shape == (20, )
    assert np.sum(mask) == 12  # 4 backbone atoms (N, CA, C, O) * 3 residues

    # Check the selected atoms are backbone atoms
    selected_atoms = small_protein[mask]
    for name in selected_atoms.atom_name:
        assert name in ["N", "CA", "C", "O"]

@pytest.mark.xfail
def test_sidechain_selection():
    """Test 'sidechain' selection."""
    mask = parser.select(small_protein, "sidechain")
    assert mask.shape == (20, )
    assert np.sum(mask) == 8  # 5 (from ALA) + 3 (from PHE) = 8 sidechain atoms

    # Check the selected atoms are not backbone atoms
    selected_atoms = small_protein[mask]
    for name in selected_atoms.atom_name:
        assert name not in ["N", "CA", "C", "O"]

def test_protein_selection():
    """Test 'protein' selection."""
    mask = parser.select(mixed_structure, "protein")
    assert mask.shape == (30, )

    # Check that only protein atoms are selected
    selected_atoms = mixed_structure[mask]
    assert np.all(selected_atoms.res_name == "TYR")

def test_nucleic_selection():
    """Test 'nucleic' selection."""
    mask = parser.select(mixed_structure, "nucleic")
    assert mask.shape == (30, )

    # Check that only nucleic atoms are selected
    selected_atoms = mixed_structure[mask]
    assert np.all(selected_atoms.res_name == "A")

def test_water_selection():
    """Test 'water' selection."""
    mask = parser.select(mixed_structure, "water")
    assert mask.shape == (30, )

    # Check that only water molecules are selected
    selected_atoms = mixed_structure[mask]
    assert np.all(selected_atoms.res_name == "HOH")

# Comparison operator tests
def test_not_equal_operator():
    """Test not equal operator."""
    mask = parser.select(small_protein, "name!=CA")
    assert mask.shape == (20, )
    assert np.sum(mask) == 17  # All atoms except CA

    # Check no CA atoms are selected
    selected_atoms = small_protein[mask]
    assert not np.any(selected_atoms.atom_name == "CA")

@pytest.mark.xfail
def test_greater_than_operator():
    """Test greater than operator."""
    mask = parser.select(small_protein, "resi>1")
    assert mask.shape == (20, )
    assert np.sum(mask) == 15  # All atoms in residues 2 and 3

    # Check only residues > 1 are selected
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_id > 1)

@pytest.mark.xfail
def test_less_than_operator():
    """Test less than operator."""
    mask = parser.select(small_protein, "resi<3")
    assert mask.shape == (20, )
    assert np.sum(mask) == 13  # All atoms in residues 1 and 2

    # Check only residues < 3 are selected
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_id < 3)

@pytest.mark.xfail
def test_greater_equal_operator():
    """Test greater than or equal operator."""
    mask = parser.select(small_protein, "resi>=2")
    assert mask.shape == (20, )
    assert np.sum(mask) == 15  # All atoms in residues 2 and 3

    # Check only residues >= 2 are selected
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_id >= 2)

@pytest.mark.xfail
def test_less_equal_operator():
    """Test less than or equal operator."""
    mask = parser.select(small_protein, "resi<=2")
    assert mask.shape == (20, )
    assert np.sum(mask) == 13  # All atoms in residues 1 and 2

    # Check only residues <= 2 are selected
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.res_id <= 2)

# Value list tests
@pytest.mark.xfail
def test_comma_separated_values():
    """Test selection with comma-separated values."""
    mask = parser.select(small_protein, "name=N,CA,C")
    assert mask.shape == (20, )
    assert np.sum(mask) == 9  # 3 residues * 3 atoms each

    # Check that only N, CA, and C atoms are selected
    selected_atoms = small_protein[mask]
    for name in selected_atoms.atom_name:
        assert name in ["N", "CA", "C"]

# Range selections
@pytest.mark.xfail
def test_residue_range():
    """Test residue range selection."""
    mask = parser.select(small_protein, "resi 1-2")
    assert mask.shape == (20, )
    assert np.sum(mask) == 13  # All atoms in residues 1 and 2

    # Check only residues in range are selected
    selected_atoms = small_protein[mask]
    for res_id in selected_atoms.res_id:
        assert 1 <= res_id <= 2

# Combination tests
@pytest.mark.xfail
def test_combine_macros_and_properties():
    """Test combining macros with property selections."""
    mask = parser.select(small_protein, "backbone and chain=A")
    assert mask.shape == (20, )

    # Check that only backbone atoms from chain A are selected
    selected_atoms = small_protein[mask]
    for name, chain in zip(selected_atoms.atom_name, selected_atoms.chain_id):
        assert name in ["N", "CA", "C", "O"]
        assert chain == "A"

# Edge cases
def test_single_atom_structure():
    """Test selection on a structure with just one atom."""
    # Create a single-atom structure
    atom_array = AtomArray(1)
    atom_array.atom_name = ["CA"]
    atom_array.res_name = ["GLY"]
    atom_array.res_id = [1]
    atom_array.chain_id = ["A"]
    atom_array.element = ["C"]
    atom_array.coord = np.zeros((1, 3))

    # Test various selections
    assert np.all(parser.select(atom_array, "all"))
    assert np.all(parser.select(atom_array, "name=CA"))
    assert not np.any(parser.select(atom_array, "name=CB"))

def test_empty_structure():
    """Test selection on an empty structure."""
    # Create an empty structure
    atom_array = AtomArray(0)

    # Test selection on empty structure
    mask = parser.select(atom_array, "all")
    assert mask.shape == (0, )

# Error handling and input validation
def test_invalid_selection_syntax():
    """Test that invalid selection syntax raises an error."""
    with pytest.raises((ValueError, Exception)):
        parser.select(small_protein, "name=CA name=CB")  # Missing operator

def test_unknown_attribute():
    """Test handling of unknown attribute."""
    with pytest.raises((ValueError, Exception)):
        parser.select(small_protein, "unknown_attr=value")

def test_mismatched_parentheses():
    """Test handling of mismatched parentheses."""
    with pytest.raises((ValueError, Exception)):
        parser.select(small_protein, "(name=CA and chain=A")

def test_invalid_numeric_comparison():
    """Test handling of invalid numeric comparison."""
    with pytest.raises((ValueError, Exception)):
        parser.select(small_protein, "resi>not_a_number")

# Complex selection tests
@pytest.mark.xfail
def test_complex_selection_chain_residue_atom():
    """Test complex selection combining chain, residue, and atom specifications."""
    mask = parser.select(small_protein, "chain=A and resi=2 and name=CA")
    assert mask.shape == (20, )
    assert np.sum(mask) == 1  # Only one atom should match

    # Verify the selected atom
    selected_atoms = small_protein[mask]
    assert selected_atoms.chain_id[0] == "A"
    assert selected_atoms.res_id[0] == 2
    assert selected_atoms.atom_name[0] == "CA"

def test_complex_selection_with_not():
    """Test complex selection with NOT operator."""
    mask = parser.select(small_protein, "chain=A and not name=CA")
    assert mask.shape == (20, )

    # Verify the selected atoms
    selected_atoms = small_protein[mask]
    assert np.all(selected_atoms.chain_id == "A")
    assert not np.any(selected_atoms.atom_name == "CA")

if __name__ == '__main__':
    main()
