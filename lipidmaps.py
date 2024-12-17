import warnings

import numpy as np
from cura.query import symbols_to_bits
from cura.store import Store
from cura.utils import mol_from_smiles
from openff.toolkit import Molecule
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    from sklearn import cluster


def store_heavy_atoms(store, tag):
    res = store.cur.execute(
        "SELECT id, smiles, natoms FROM molecules WHERE tag = ?", (tag,)
    )
    data = []
    for id, smiles, natoms in tqdm(res.fetchall()):
        mol = mol_from_smiles(smiles)
        natoms_ = mol.GetNumAtoms()
        nheavy = mol.GetNumHeavyAtoms()
        assert natoms == natoms_

        data.append((nheavy, id))

    store.cur.executemany(
        """
    UPDATE molecules
    SET nheavy = ?
    WHERE id = ?
    """,
        data,
    )
    store.con.commit()


def store_fragment_heavy_atoms(store, tag):
    res = store.cur.execute(
        "SELECT id, smiles, natoms FROM fragments WHERE tag = ?", (tag,)
    )
    data = []
    for id, smiles, natoms in tqdm(res.fetchall()):
        mol = mol_from_smiles(smiles)
        natoms_ = mol.GetNumAtoms()
        nheavy = mol.GetNumHeavyAtoms()
        assert natoms == natoms_

        data.append((nheavy, id))

    store.cur.executemany(
        """
    UPDATE fragments
    SET nheavy = ?
    WHERE id = ?
    """,
        data,
    )
    store.con.commit()


# 15:04 to run, definitely seems faster using executemany
def store_fragment_fingerprints(store, tag):
    res = store.cur.execute(
        "SELECT id, smiles FROM fragments WHERE tag = ?", (tag,)
    )
    fpgen = AllChem.GetMorganGenerator()

    data = []
    for id, smiles in tqdm(res.fetchall()):
        mol = mol_from_smiles(smiles)
        fp = fpgen.GetFingerprint(mol)
        data.append((fp.ToBitString(), id))

    store.cur.executemany(
        """
    UPDATE fragments
    SET morgan = ?
    WHERE id = ?
    """,
        data,
    )
    store.con.commit()


# 12:39 to run, probably faster with executemany and not committing each time
def store_fingerprints(store):
    res = store.cur.execute(
        "SELECT id, smiles FROM molecules WHERE tag = ?", (tag,)
    )
    fpgen = AllChem.GetMorganGenerator()

    for id, smiles in tqdm(res.fetchall()):
        mol = mol_from_smiles(smiles)
        fp = fpgen.GetFingerprint(mol)
        store.cur.execute(
            """
        UPDATE molecules
        SET morgan = ?
        WHERE id = ?
        """,
            (fp.ToBitString(), id),
        )
        store.con.commit()


def load_fragment_fingerprints(store, tag, max_nheavy, min_nheavy, elements):
    res = store.cur.execute(
        """
        SELECT id, morgan, elements
        FROM fragments
        WHERE tag = ?
        AND nheavy < ?
        AND nheavy > ?
        """,
        (tag, max_nheavy, min_nheavy),
    )
    # element filtering adapted from cura ElementFilter and Store.get_molecules
    # (for int conversion)
    mask = symbols_to_bits(elements)

    ret = [
        (id, DataStructs.CreateFromBitString(fp))
        for id, fp, elements in res.fetchall()
        if (int.from_bytes(elements, "big") | mask) == mask
    ]

    ids, fps = zip(*ret)
    return ids, fps


# this takes 1.26 seconds, so very nice to load fingerprints from the database
def load_fingerprints(store, tag, max_nheavy, min_nheavy, elements):
    res = store.cur.execute(
        """
        SELECT id, morgan, elements
        FROM molecules
        WHERE tag = ?
        AND nheavy < ?
        AND nheavy > ?
        """,
        (tag, max_nheavy, min_nheavy),
    )

    # element filtering adapted from cura ElementFilter and Store.get_molecules
    # (for int conversion)
    mask = symbols_to_bits(elements)

    ret = [
        (id, DataStructs.CreateFromBitString(fp))
        for id, fp, elements in res.fetchall()
        if (int.from_bytes(elements, "big") | mask) == mask
    ]

    ids, fps = zip(*ret)
    return ids, fps


def get_smiles(store, mol_id, table="fragments"):
    res = store.cur.execute(
        f"SELECT smiles FROM {table} WHERE id = ?", (mol_id,)
    )
    return res.fetchone()[0]


def tanimoto(a1, a2):
    return 1 - np.bitwise_and(a1, a2).sum() / np.bitwise_or(a1, a2).sum()


def try_dbscan(fps):
    dbscan = cluster.DBSCAN(metric=tanimoto)

    # running for 35 minutes with no output, which I guess is expected. at
    # least it's using a fairly constant amount of memory
    n = 5000

    print(f"expecting {3.48e-6 * n**2 - 2.72e-4 + 2:.2} sec")

    labels = dbscan.fit_predict(fps[:n])

    # pretty much O(x^2) with coefficient 3e-6

    # 100 in 1.9 sec
    # 1000 in 5.49 sec
    # 2000 in 15.1 sec
    # 3000 in 32.6 sec
    # 5000 in 80 sec

    print(labels)

    # took 3 minutes and used a ton of RAM
    # t = tanimoto(fps)

    # bs = fps[0].ToBitString()

    # we can round trip through a bitstring saved as text in the db. sqlite has
    # stuff like varchar(n) and character(n) but only has one internal text
    # representation, so it doesn't matter to try to specify the 2048 length.
    # just store as text

    # print(len(bs))
    # bs2 = DataStructs.CreateFromBitString(bs)

    return None  # nice for folding


def replace_dummies(mol: Chem.Mol) -> Chem.Mol:
    """Replace dummy atoms in ``mol`` with hydrogen.

    Also handle a special case where S(=O)(=O)* is replaced with a S(=O)([O-]).

    Adapted from some fragmentation code I got from Lily.
    """
    rd_dummy_replacements = [
        # Handle the special case of -S(=O)(=O)[*] -> -S(=O)(-[O-])
        (Chem.MolFromSmiles("S(=O)(=O)*"), Chem.MolFromSmiles("S(=O)([O-])")),
        # Handle the general case
        (Chem.MolFromSmiles("*"), Chem.MolFromSmiles("[H]")),
    ]

    for pat, rep in rd_dummy_replacements:
        mol = AllChem.ReplaceSubstructs(mol, pat, rep, replaceAll=True)[0]

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    return mol


store = Store("store.sqlite")
tag = "data/lipidmaps.smi"

# reusing inchi list from vf/02/frag/inchis.dat. should contain all the inchis
# from the opt and td datasets I've been using for refitting tm, plus the
# industry dataset
with open("inchis.dat") as inp:
    old_inchis = {inchi.strip() for inchi in inp}

# trying LazyBitVectorPick from rdkit:
# https://greglandrum.github.io/rdkit-blog/posts/2020-11-18-sphere-exclusion-clustering.html

lp = rdSimDivPickers.LeaderPicker()

ids, fps = load_fingerprints(
    store,
    tag,
    max_nheavy=100,
    min_nheavy=3,
    # include X to handle fragments
    elements=["Cl", "P", "Br", "I", "H", "C", "O", "N", "F", "S", "X"],
)
# 295745 before element filter, 295556 after
print(f"processing {len(fps)} fingerprints")
# slightly strange threshold gives 4003 smiles after deduplication and the
# inchi filter
thresh = 0.705
picks = lp.LazyBitVectorPick(fps, len(fps), thresh)
print(f"found {len(picks)} clusters")

fragments = (mol_from_smiles(get_smiles(store, ids[p])) for p in picks)

smiles = set()
for frag in tqdm(fragments, total=len(picks)):
    m = replace_dummies(frag)
    s = Chem.MolToSmiles(m)
    inchi = Molecule.from_smiles(s, allow_undefined_stereo=True).to_inchikey()
    if inchi not in old_inchis:
        smiles.add(s)

print(f"found {len(smiles)} deduplicated smiles")

with open("mols.smi", "w") as out:
    for s in sorted(smiles, key=lambda s: len(s)):
        print(s, file=out)
