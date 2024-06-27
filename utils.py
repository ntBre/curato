import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage, rdDepictor, rdMolDraw2D


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Create an RDKit molecule from SMILES and perform the cleaning operations
    from the OpenFF toolkit

    """
    rdmol = Chem.MolFromSmiles(smiles)
    return openff_clean(rdmol)


def mol_from_smarts(smarts: str) -> Chem.Mol:
    """Create an RDKit molecule from SMARTS and perform the cleaning operations
    from the OpenFF toolkit

    """
    rdmol = Chem.MolFromSmarts(smarts)
    return openff_clean(rdmol)


def openff_clean(rdmol: Chem.Mol) -> Chem.Mol:
    Chem.SanitizeMol(
        rdmol,
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.AssignStereochemistry(rdmol)
    rdmol = Chem.AddHs(rdmol)
    return rdmol


def find_smarts_matches(mol, smirks: Chem.Mol):
    "Adapted from RDKitToolkitWrapper._find_smarts_matches"
    idx_map = dict()
    for atom in smirks.GetAtoms():
        smirks_index = atom.GetAtomMapNum()
        if smirks_index != 0:
            idx_map[smirks_index - 1] = atom.GetIdx()
    map_list = [idx_map[x] for x in sorted(idx_map)]

    max_matches = np.iinfo(np.uintc).max
    match_kwargs = dict(
        uniquify=False, maxMatches=max_matches, useChirality=True
    )
    full_matches = mol.GetSubstructMatches(smirks, **match_kwargs)

    matches = [tuple(match[x] for x in map_list) for match in full_matches]

    return matches


def find_matches(
    params: list[tuple[str, Chem.Mol]], mol: Chem.Mol
) -> dict[tuple[int], str]:
    """Returns a map of chemical environment tuples to their matching parameter
    ids. Modeled (loosely) after ForceField.label_molecules and
    ParameterHandler._find_matches"""
    matches = {}
    for id, smirks in params:
        env_matches = find_smarts_matches(mol, smirks)
        for mat in env_matches:
            if mat[0] > mat[-1]:
                mat = mat[::-1]
            matches[mat] = id
    return matches


def make_svg(pid, map, mol_map, mol) -> tuple[list[str], list[list[int]]]:
    """Returns a sequence of SVGs corresponding to each match for `pid` in
    `mol`, along with the corresponding list of chemical environment tuples.

    """
    hl_atoms = []
    if pid in map:
        tmp = find_matches(mol_map, mol)
        hl_atoms = [
            atoms for atoms, param_id in tmp.items() if param_id == pid
        ]
    return mol_to_svg(mol, 400, 300, "", hl_atoms), hl_atoms


def mol_to_svg(
    mol: Chem.Mol,
    width: int,
    height: int,
    legend: str,
    hl_atoms: list[list[int]],
) -> list[str]:
    "Return a list of SVGs, one for each match in hl_atoms"

    if len(hl_atoms) == 0:
        hl_atoms = [[]]

    ret = []
    for hl in hl_atoms:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        rdmol = rdMolDraw2D.PrepareMolForDrawing(mol)
        # PrepareMolForDrawing returns a copy, and I asserted that `mol` still
        # had mapping numbers after doing this
        for atom in rdmol.GetAtoms():
            atom.SetAtomMapNum(0)

        ret.append(
            MolsToGridImage(
                [rdmol],
                useSVG=True,
                highlightAtomLists=[hl],
                subImgSize=(300, 300),
                molsPerRow=1,
            )
        )
    return ret


def find_smallest(mols: list[Chem.Mol], cluster: list[int]) -> int:
    "Return the index of the smallest molecule in `cluster`"
    tmp = [(i, mols[c]) for i, c in enumerate(cluster)]
    return min(tmp, key=lambda x: x[1].GetNumAtoms())[0]


def mol_to_smiles(mol: Chem.Mol, mapped=False) -> str:
    "Very simplified version of offtk rdkit_wrapper to_smiles"

    # this should check for an existing atom map, such as the one provided by
    # mol_from_mapped_smiles
    if mapped and any((a.GetAtomMapNum() == 0 for a in mol.GetAtoms())):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


def mol_from_mapped_smiles(smiles) -> Chem.Mol:
    """Adapted from offtk rdkit_wrapper from_smiles without skipping the atom
    mapping part like `mol_from_smiles`"""
    rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
    # the toolkit says we have to strip the atom map to avoid affecting the
    # stereochemistry tags. I'm stripping it out for the call to `openff_clean`
    # and then putting it back. hopefully that works
    for atom in rdmol.GetAtoms():
        if (map_num := atom.GetAtomMapNum()) != 0:
            atom.SetProp("_map_idx", str(map_num))
            atom.SetAtomMapNum(0)
    ret = openff_clean(rdmol)
    for atom in ret.GetAtoms():
        map_num = atom.GetProp("_map_idx")
        atom.SetAtomMapNum(int(map_num))
    return ret


def into_params(ff) -> list[tuple[str, Chem.Mol]]:
    "Convert a ForceField into a sequence of pid:Mol pairs"
    return [
        (p.id, Chem.MolFromSmarts(p.smirks))
        for p in ff.get_parameter_handler("ProperTorsions").parameters
    ]
