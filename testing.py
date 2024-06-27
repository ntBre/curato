import tempfile

from openff.toolkit import ForceField, Molecule, RDKitToolkitWrapper
from openff.toolkit.utils import ToolkitRegistry, toolkit_registry_manager
from rdkit import Chem

from query import load_want
from store import (
    DBMol,
    Store,
    bits_to_elements,
    elements_to_bits,
    get_elements,
)
from utils import (
    find_matches,
    into_params,
    mol_from_mapped_smiles,
    mol_from_smiles,
    mol_to_smiles,
)


def test_find_matches():
    ff = ForceField("openff-2.1.0.offxml")
    params = into_params(ff)

    for smile in ["CCO"]:
        offmol = Molecule.from_smiles(smile)
        rdmol = mol_from_smiles(smile)
        got = list(sorted(find_matches(params, rdmol).items()))
        want = list(
            (k, v.id)
            for k, v in sorted(
                ff.label_molecules(offmol.to_topology())[0][
                    "ProperTorsions"
                ].items()
            )
        )
        assert got == want, f"got = {got}, want = {want}"


def test_load_want():
    got = len(load_want("want.params"))
    want = 62
    assert got == want


def test_elements_to_bits():
    mol = Chem.MolFromSmiles("CCO")
    got = elements_to_bits(get_elements(mol))
    want = 0b101000000
    assert got == want

    got = bits_to_elements(got)
    want = [6, 8]
    assert got == want


def test_store():
    with tempfile.NamedTemporaryFile() as f:
        s = Store(f.name)
        mol = DBMol("CC[At]", "inchikey", 3, 1 << 85 | 1 << 6)
        s.insert_molecule(mol)
        got = next(s.get_molecules()).get_elements()
        want = [6, 85]
        assert got == want


def test_query():
    from query import _main

    ffname = "openff-2.1.0.offxml"
    with tempfile.NamedTemporaryFile() as f:
        s = Store(f.name)
        s.insert_molecules(
            [
                DBMol("CCO", "inchikey", 3, 1 << 6 | 1 << 8),
            ]
        )
        s.insert_fragments([DBMol("*CCO", None, 3, 1 << 6 | 1 << 8)])
        s, got = _main(8, 32, [], f.name, ffname, {"t1", "t2", "t9"}, 100)
        want = s.get_forcefield(ffname)
        assert got == want.matches


def test_mol_to_smiles():
    smiles = "CCO"
    rdk = mol_from_smiles(smiles)
    off = Molecule.from_smiles(smiles)
    got = mol_to_smiles(rdk, mapped=True)
    with toolkit_registry_manager(ToolkitRegistry([RDKitToolkitWrapper()])):
        want = off.to_smiles(mapped=True)
    assert got == want


def test_mapping():
    smiles = "[H:5][N+:1]([H:6])([H:7])[S:2](=[O:3])[O:4][H:8]"
    with toolkit_registry_manager(ToolkitRegistry([RDKitToolkitWrapper()])):
        off = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        want = off.to_smiles(mapped=True)
    rdk = mol_from_mapped_smiles(smiles)
    got = mol_to_smiles(rdk, mapped=True)

    assert got == want


def test_mapping_debug():
    smiles = (
        "[C:1]([O:2][C:3](=[O:4])[C:5]([C:6]([N+:7]1([O-:8])"
        "[C:9]([H:18])([H:19])[C:10]1([H:20])[H:21])([H:16])"
        "[H:17])([H:14])[H:15])([H:11])([H:12])[H:13]"
    )
    mol = Molecule.from_mapped_smiles(smiles)
    ff = ForceField(
        "../../01_generate-forcefield/output/"
        "initial-force-field-openff-2.1.0.offxml"
    )
    want = list(
        (k, v.id)
        for k, v in sorted(
            ff.label_molecules(mol.to_topology())[0]["ProperTorsions"].items()
        )
    )

    params = into_params(ff)
    mol = mol_from_mapped_smiles(smiles)
    pairs = find_matches(params, mol).items()
    # find_matches returns atom indices, which in the case of my
    # `mol_from_mapped_smiles` may be (and likely are) different from the atom
    # map. ff.label_molecules also returns atom indices, but the toolkit
    # ensures that they correspond to the atom map (minus 1). this code maps
    # from the native rdkit indices to the mapping to match the toolkit output.
    # I originally made this an option to pass to find_matches, but I think
    # it's easier to do it after the fact only when necessary
    map_to_id = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    got = []
    for k, v in pairs:
        tup, v = (tuple([map_to_id[a] - 1 for a in k]), v)
        if tup[0] > tup[-1]:
            tup = tup[::-1]
        got.append((tup, v))
    got = list(sorted(got))
    print(got)

    print(len(got))
    assert len(got) == len(want)
    assert got == want
