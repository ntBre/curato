import pickle
import sqlite3
from itertools import chain
from multiprocessing import Pool
from typing import Iterator

import click
from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from cut_compound import Compound
from utils import mol_from_smiles


class DBMol:
    "Representation of a molecule in the database"
    id: int
    smiles: str
    inchikey: str
    natoms: int
    elements: int
    tag: str

    def __init__(self, smiles, inchikey, natoms, elements, tag=None, id=None):
        self.smiles = smiles
        self.inchikey = inchikey
        self.natoms = natoms
        self.elements = elements
        self.tag = tag
        self.id = id

    def __repr__(self):
        return (
            f"DBMol(id={self.id}, smiles={self.smiles}, natoms={self.natoms})"
        )

    @classmethod
    def from_rdmol(cls, mol: Chem.Mol, fragment: bool):
        if fragment:
            inchi = None
        else:
            inchi = Chem.MolToInchiKey(mol)
        return cls(
            smiles=Chem.MolToSmiles(mol),
            natoms=mol.GetNumAtoms(),
            elements=elements_to_bits(get_elements(mol)),
            inchikey=inchi,
        )

    def get_elements(self) -> list[int]:
        "Return `self.elements` as a list of atomic numbers"
        return bits_to_elements(self.elements)


class Match:
    smirks: str
    pid: str
    molecules: list[str]  # TODO this is actually supposed to be list[rowid]
    fragments: list[str]  # TODO this is actually supposed to be list[rowid]

    def __init__(self, smirks, pid, molecules, fragments):
        self.smirks = smirks
        self.pid = pid
        self.molecules = molecules
        self.fragments = fragments

    def __eq__(self, o):
        return (
            self.smirks == o.smirks
            and self.pid == o.pid
            and self.molecules == o.molecules
            and self.fragments == o.fragments
        )

    def to_dict(self):
        return dict(smirks=self.smirks, pid=self.pid, molecules=self.molecules)


class DBForceField:
    id: int
    name: str
    matches: list[Match]

    def __init__(self, name, matches, id=None):
        self.id = id
        self.name = name
        self.matches = matches


class Store:
    def __init__(self, filename="store.sqlite", nprocs=8):
        self.con = sqlite3.connect(filename)
        self.cur = self.con.cursor()
        self.cur.arraysize = 1024
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS molecules (
            id integer primary key,
            smiles text unique,
            inchikey text unique,
            natoms int,
            elements blob,
            tag text
            )
            """
        )
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS fragments (
            id integer primary key,
            smiles text unique,
            natoms int,
            elements blob,
            tag text
            )
            """
        )
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS forcefields (
            id integer primary key,
            name text unique,
            matches blob
            )
            """
        )
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS dataset (
            id integer primary key,
            smiles text,
            pid text,
            hl_atoms blob,
            CONSTRAINT unq UNIQUE (smiles, pid, hl_atoms)
            )
            """
        )
        self.nprocs = nprocs

    @classmethod
    def quick(cls, filename="store.sqlite", nprocs=8):
        """Connect to an existing `Store` without attempting to initialize the
        tables.

        This is intended for temporary use such as in `serve.py`.

        """
        self = cls.__new__(cls)
        self.con = sqlite3.connect(filename)
        self.cur = self.con.cursor()
        self.cur.arraysize = 1024
        self.nprocs = nprocs
        return self

    def _insert_dbmols(self, mols: list[DBMol], tblname: str):
        self.cur.executemany(
            f"""INSERT OR IGNORE INTO {tblname} (smiles, natoms, elements, tag)
            VALUES (?1, ?2, ?3, ?4)""",
            [
                (m.smiles, m.natoms, m.elements.to_bytes(128, "big"), m.tag)
                for m in mols
            ],
        )
        self.con.commit()

    def insert_molecule(self, mol: DBMol):
        "Insert a single SMILES into the database"
        self.insert_molecules([mol])

    def insert_molecules(self, mols: list[DBMol]):
        "Insert multiple SMILES into the molecules table"
        self.cur.executemany(
            """INSERT OR IGNORE INTO molecules
            (smiles, inchikey, natoms, elements, tag)
            VALUES (?1, ?2, ?3, ?4, ?5)""",
            [
                (
                    m.smiles,
                    m.inchikey,
                    m.natoms,
                    m.elements.to_bytes(128, "big"),
                    m.tag,
                )
                for m in mols
            ],
        )
        self.con.commit()

    def insert_fragments(self, mols: list[DBMol]):
        "Insert multiple SMILES into the fragments table"
        self._insert_dbmols(mols, "fragments")

    def insert_forcefield(self, ff: DBForceField):
        self.cur.execute(
            """INSERT OR IGNORE INTO forcefields (name, matches)
            VALUES (?1, ?2)""",
            (ff.name, sqlite3.Binary(pickle.dumps(ff.matches))),
        )
        self.con.commit()

    def get_forcefield(self, ffname: str) -> DBForceField:
        id, name, matches = self.cur.execute(
            "SELECT id, name, matches FROM forcefields WHERE name = ?1",
            (ffname,),
        ).fetchone()
        return DBForceField(id=id, name=name, matches=pickle.loads(matches))

    def reset_forcefield(self, ffname: str):
        self.cur.execute("DELETE FROM forcefields WHERE name = ?1", (ffname,))
        self.con.commit()

    def get_smiles_matching(self, ffname: str, pid: str) -> list[str]:
        "Return the SMILES matching `pid` for `ffname`"
        dbff = self.get_forcefield(ffname)
        smiles = []
        for m in dbff.matches:
            if m.pid == pid:
                smiles.extend(m.molecules)
        return smiles

    def get_sizehint(self) -> int:
        "Return a count of rows in the database"
        return self.cur.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]

    def get_smiles(self, limit=None) -> Iterator[str]:
        "Return an iterator over SMILES in the database"
        if limit:
            res = self.cur.execute(
                "SELECT smiles FROM molecules limit ?", (limit,)
            )
        else:
            res = self.cur.execute("SELECT smiles FROM molecules")
        while len(v := res.fetchmany()) > 0:
            # unpack 1-tuples
            yield from (x[0] for x in v)

    def get_molecules(self, limit=None) -> Iterator[DBMol]:
        if limit:
            res = self.cur.execute(
                """SELECT id, smiles, inchikey, natoms, elements, tag
                FROM molecules
                limit ?""",
                (limit,),
            )
        else:
            res = self.cur.execute(
                """SELECT id, smiles, inchikey, natoms, elements, tag
                FROM molecules"""
            )
        while len(v := res.fetchmany()) > 0:
            # unpack 3-tuples
            yield from (
                DBMol(
                    id=x[0],
                    smiles=x[1],
                    inchikey=[2],
                    natoms=x[3],
                    elements=int.from_bytes(x[4], "big"),
                    tag=x[5],
                )
                for x in v
            )

    def get_fragments(self, limit=None) -> Iterator[DBMol]:
        return self._get_dbmols("fragments", limit)

    def _get_dbmols(self, tablename: str, limit=None) -> Iterator[DBMol]:
        if limit:
            res = self.cur.execute(
                f"""SELECT id, smiles, natoms, elements, tag FROM {tablename}
                limit ?""",
                (limit,),
            )
        else:
            res = self.cur.execute(
                f"SELECT id, smiles, natoms, elements, tag FROM {tablename}"
            )
        while len(v := res.fetchmany()) > 0:
            # unpack 3-tuples
            yield from (
                DBMol(
                    id=x[0],
                    smiles=x[1],
                    natoms=x[2],
                    elements=int.from_bytes(x[3], "big"),
                    tag=x[4],
                    inchikey=None,
                )
                for x in v
            )

    def add_to_dataset(self, smiles: str, pid: str, hl_atoms: tuple[int]):
        self.cur.execute(
            """INSERT OR IGNORE INTO dataset (smiles, pid, hl_atoms)
            VALUES (?1, ?2, ?3)
            """,
            (smiles, pid, sqlite3.Binary(pickle.dumps(tuple(hl_atoms)))),
        )
        self.con.commit()

    def get_dataset_entries(
        self,
    ) -> Iterator[tuple[int, str, str, tuple[int]]]:
        res = self.cur.execute("SELECT id, smiles, pid, hl_atoms FROM dataset")
        while len(v := res.fetchmany()) > 0:
            yield from (
                (id, smiles, pid, tuple(pickle.loads(hl_atoms)))
                for id, smiles, pid, hl_atoms in v
            )

    def get_dataset_size(self) -> int:
        res = self.cur.execute("SELECT COUNT(*) from dataset")
        return res.fetchone()[0]

    def reset_dataset(self):
        self.cur.execute("DELETE FROM dataset")
        self.con.commit()

    def process_line(line) -> tuple[list[DBMol], list[DBMol]]:
        "Returns a list of fragments and a list of whole molecules"
        [_chembl_id, cmiles, _inchi, _inchikey] = line.split("\t")
        all_smiles = cmiles.split(".")
        frags = list()
        mols = list()
        for smiles in all_smiles:
            try:
                rdmol = mol_from_smiles(smiles)
            except Exception as e:
                print(f"failed to build rdmol with: `{e}`")
                continue
            if Descriptors.NumRadicalElectrons(rdmol) > 0:
                continue
            mols.append(DBMol.from_rdmol(rdmol, fragment=False))
            if x := xff(rdmol):
                frags.extend(x)
        return frags, mols

    def load_chembl(self, filename) -> dict[str, Molecule]:
        with open(filename) as inp, Pool(processes=self.nprocs) as pool:
            for frags, mols in tqdm(
                pool.imap_unordered(
                    Store.process_line,
                    (line for i, line in enumerate(inp) if i > 0),
                    chunksize=32,
                ),
                total=2372675,
            ):
                if frags:
                    for frag in frags:
                        frag.tag = filename
                    self.insert_fragments(frags)
                    self.insert_molecules(mols)


def find_frag_bonds(rdmol, keep_atoms):
    "Locate bonds between atoms to keep and those to remove"
    to_remove = []
    for bond in rdmol.GetBonds():
        b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (b1 in keep_atoms and b2 not in keep_atoms) or (
            b2 in keep_atoms and b1 not in keep_atoms
        ):
            to_remove.append(rdmol.GetBondBetweenAtoms(b1, b2).GetIdx())
    return to_remove


def xff(mol: Chem.Mol) -> list[DBMol] | None:
    c = Compound(mol)
    frags = c.cutCompound()

    ret = list()
    for fr in chain(frags.frag_rings, frags.frag_chains):
        emol = Chem.Mol(c.rdmol)
        frag_atoms = set(fr)
        rdbonds = find_frag_bonds(emol, frag_atoms)
        if rdbonds:
            fragmented = Chem.FragmentOnBonds(
                emol,
                rdbonds,
                addDummies=True,
                dummyLabels=[(0, 0)] * len(rdbonds),
            )
            for frag in Chem.GetMolFrags(fragmented, asMols=True):
                ret.append(DBMol.from_rdmol(frag, fragment=True))

    return ret


def get_elements(mol: Chem.Mol) -> list[int]:
    "Return a list of atomic numbers in `mol`"
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def elements_to_bits(nums: list[int]) -> int:
    "Pack a sequence of atomic numbers into an int"
    ret = 0
    for n in nums:
        ret |= 1 << n
    return ret


def bits_to_elements(bits: int) -> list[int]:
    "Unpack an int into a sequence of atomic numbers"
    ret = []
    for i in range(128):
        if (bits & (1 << i)) != 0:
            ret.append(i)
    return ret


@click.command()
@click.option("--nprocs", "-n", type=int, default=8)
def main(nprocs):
    store = Store(nprocs=nprocs)
    store.load_chembl("chembl_33_chemreps.txt")


if __name__ == "__main__":
    main()
