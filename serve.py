import logging
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus

import numpy as np
from flask import Flask, redirect, request, send_from_directory, url_for
from jinja2 import Environment, PackageLoader, select_autoescape
from openff.toolkit import ForceField
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D

from parse import tanimoto
from query import PTABLE, find_matches, into_params, mol_from_smiles
from store import Store
from utils import (
    find_smallest,
    make_svg,
    mol_from_mapped_smiles,
    mol_to_smiles,
    mol_to_svg,
    openff_clean,
)

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    from sklearn import cluster as skcluster

logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG)

app = Flask("serve")
env = Environment(
    loader=PackageLoader("serve"), autoescape=select_autoescape()
)
env.globals["Chem"] = Chem
env.globals["make_svg"] = make_svg
env.globals["find_smallest"] = find_smallest
env.globals["mol_to_smiles"] = mol_to_smiles
env.globals["list"] = list
dbname = "store.sqlite"
ffname = (
    "../../01_generate-forcefield/output/"
    "initial-force-field-openff-2.1.0.offxml"
)
off = ForceField(ffname)
pid_to_smarts = {
    p.id: p.smirks
    for p in off.get_parameter_handler("ProperTorsions").parameters
}
mol_map = into_params(off)


PID_RE = re.compile("(t)([0-9]+)(.*)")


def pid_sort(pid: str) -> tuple[str, int, str | None]:
    "Return the fields of a ProperTorsion parameter ID as a tuple for sorting"
    t, n, tail = PID_RE.match(pid).groups()
    return (t, int(n), tail)


@app.route("/js/<path:name>")
def js(name):
    return send_from_directory("js", name)


@app.route("/css/<path:name>")
def css(name):
    return send_from_directory("css", name)


@app.route("/")
def index():
    template = env.get_template("index.html")
    table = Store.quick()
    pairs = [
        (m.pid, len(m.molecules)) for m in table.get_forcefield(ffname).matches
    ]
    pairs = sorted(pairs, key=lambda x: pid_sort(x[0]))
    parameter_ids, molecule_counts = (
        [x[0] for x in pairs],
        [x[1] for x in pairs],
    )

    dsmap = defaultdict(int)
    for _id, _smiles, pid, _hl_atoms in table.get_dataset_entries():
        dsmap[pid] += 1

    pid_counts = []
    for pid in parameter_ids:
        pid_counts.append(dsmap.get(pid, 0))

    ds_size = table.get_dataset_size()

    return template.render(
        parameter_ids=parameter_ids,
        molecule_counts=molecule_counts,
        pid_counts=pid_counts,
        ds_size=ds_size,
    )


@dataclass
class DrawMol:
    smiles: str
    natoms: int
    svgs: list[str]
    hl_atoms: list[list[int]]


@app.route("/param/<pid>")
def param(pid):
    MAX_DRAW = 50
    max_draw = int(request.args.get("max", MAX_DRAW))
    table = Store.quick()
    mols = get_smiles_list(table, ffname, pid)

    draw_mols = mols_to_draw(mols, pid, mol_map, max_draw)

    smarts = pid_to_smarts[pid]  # TODO error page on missing
    total_mols = len(mols)
    template = env.get_template("param.html")
    return template.render(
        smarts=smarts,
        pid=pid,
        mols=draw_mols,
        cur_mols=len(draw_mols),
        total_mols=total_mols,
    )


def get_smiles_list(table, ffname, pid) -> list[tuple[Chem.Mol, str, int]]:
    """Return a sequence of Mol, SMILES, natom tuples from `table`, sorted by
    natoms.

    """
    smiles_list = table.get_smiles_matching(ffname, pid)
    mols = []
    for s in smiles_list:
        mol = mol_from_smiles(s)
        natoms = mol.GetNumAtoms()
        mols.append((mol, s, natoms))
    mols = sorted(mols, key=lambda k: k[2])  # sort by natoms
    return mols


def mol_to_draw(mol, pid, natoms, atoms=None):
    matches = find_matches(mol_map, mol)
    hl_atoms = []
    for _atoms, mpid in matches.items():
        if mpid == pid:
            hl_atoms.append(_atoms)
    if atoms:
        # find_matches returns atom indices, not atom map nums because it was
        # constructed from the original smiles where these two were the same.
        # here, I convert these original indices (now captured in the mapped
        # smiles) to indices in the current molecule
        map_to_id = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
        atoms = tuple([map_to_id[a + 1] for a in atoms])
        if atoms[0] > atoms[-1]:
            atoms = atoms[::-1]
        if atoms in hl_atoms:
            hl_atoms = [atoms]
        else:
            logger.warn(f"requested atoms {atoms} not found in {hl_atoms}")
            hl_atoms = []
    svg = mol_to_svg(mol, 300, 300, "", hl_atoms)
    smiles = mol_to_smiles(mol, mapped=True)
    return DrawMol(smiles, natoms, svg, hl_atoms)


def mols_to_draw(mols, pid, mol_map, max_draw):
    draw_mols = []
    for mol, smiles, natoms in mols[:max_draw]:
        draw_mols.append(mol_to_draw(mol, pid, natoms))
    return draw_mols


@dataclass
class Report:
    max: int
    nfps: int
    noise: int
    clusters: list[list[int]]
    mols: list[Chem.Mol]
    map: dict[str, str]
    mol_map: list[tuple[str, Chem.Mol]]


def make_fps(mols: list[Chem.Mol], radius: int):
    fpgen = Chem.AllChem.GetMorganGenerator(radius=radius)
    return [fpgen.GetFingerprint(mol) for mol in mols]


def dbscan(dist, eps, min_pts):
    db = skcluster.DBSCAN(eps=eps, min_samples=min_pts)
    return db.fit(dist).labels_


def make_cluster_report(ffname, mols, eps, min_pts) -> Report:
    MORGAN_RADIUS = 4
    map = {
        p.id: p.smirks
        for p in off.get_parameter_handler("ProperTorsions").parameters
    }
    mol_map = into_params(off)
    fps = make_fps(mols, MORGAN_RADIUS)
    nfps = len(fps)
    dist = tanimoto(fps)
    labels = dbscan(dist, eps, min_pts)
    label_max = max(labels)
    clusters = [[] for _ in range(label_max + 1)]
    noise = []
    for i, el in enumerate(labels):
        if el < 0:
            noise.append(i)
        else:
            clusters[el].append(i)

    noise_pts = len(noise)
    if len(clusters[0]) == 0:
        print(f"warning: all noise points: {labels}")
        clusters[0] = noise

    return Report(
        max=label_max,
        nfps=nfps,
        noise=noise_pts,
        clusters=clusters,
        mols=mols,
        map=map,
        mol_map=mol_map,
    )


@app.route("/cluster/<pid>")
def cluster(pid):
    smarts = pid_to_smarts[pid]
    table = Store.quick()
    mols = [mol_from_smiles(s) for s in table.get_smiles_matching(ffname, pid)]
    DBSCAN_EPS = 0.5
    DBSCAN_MIN_PTS = 1

    eps = request.args.get("eps", default=DBSCAN_EPS, type=float)
    min_pts = request.args.get("min_pts", default=DBSCAN_MIN_PTS, type=int)

    start = time.time()
    r = make_cluster_report(ffname, mols, eps, min_pts)
    end = time.time()

    elapsed = f"{end - start:.2f} sec"

    clusters = sorted(r.clusters, key=lambda c: r.mols[c[0]].GetNumAtoms())
    template = env.get_template("cluster.html")
    return template.render(
        pid=pid,
        smarts=smarts,
        eps=eps,
        min_pts=min_pts,
        max=r.max,
        nfps=r.nfps,
        noise=r.noise,
        clusters=clusters,
        mols=r.mols,
        map=r.map,
        mol_map=r.mol_map,
        elapsed=elapsed,
    )


@app.route("/add-molecule", methods=["POST"])
def add_molecule():
    data = request.get_json()
    table = Store.quick()
    table.add_to_dataset(data["smiles"], data["pid"], data["hl_atoms"])
    return "", HTTPStatus.CREATED


# the current rdkit molecule being edited. shared between `edit_molecule` and
# `update_molecule`
CUR_EDIT_MOL: Chem.Mol = None


@dataclass
class Atom:
    index: int
    symbol: str
    charge: int


@dataclass
class Bond:
    index: int
    atom1: int
    atom2: int
    order: int


CANVAS_SIZE = 400  # size of html canvas


@app.route("/edit-molecule", methods=["POST"])
def edit_molecule():
    data = request.get_json()
    smiles, pid = data["smiles"], data["pid"]
    mol = mol_from_smiles(smiles)
    mol, ret = mol_to_js(mol, pid)
    global CUR_EDIT_MOL
    CUR_EDIT_MOL = mol
    return ret, HTTPStatus.CREATED


def mol_to_js(mol, pid):
    matches = find_matches(mol_map, mol)
    hl_atoms = []
    for atoms, mpid in matches.items():
        if mpid == pid:
            hl_atoms = atoms
            break
    hl_atoms = [] if hl_atoms is None else hl_atoms
    rdDepictor.SetPreferCoordGen(True)
    # this seems to help with a segfault I was having before
    mol = mol_from_smiles(Chem.MolToSmiles(mol))
    rdDepictor.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    atoms = [
        Atom(
            atom.GetIdx(), PTABLE[atom.GetAtomicNum()], atom.GetFormalCharge()
        )
        for atom in mol.GetAtoms()
    ]
    bonds = [
        Bond(
            b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()
        )
        for b in mol.GetBonds()
    ]
    coords = conf.GetPositions()
    xs, ys = coords[:, 0], coords[:, 1]
    minx, maxx = np.min(xs), np.max(xs)
    miny, maxy = np.min(ys), np.max(ys)
    xs -= minx
    ys -= miny
    # normalize the coordinates in the middle 3/4 of the canvas to prevent
    # overflow
    xs *= 0.750 * CANVAS_SIZE / (maxx - minx)
    ys *= 0.750 * CANVAS_SIZE / (maxy - miny)
    xs += 0.125 * CANVAS_SIZE
    ys += 0.125 * CANVAS_SIZE
    return mol, dict(
        atoms=atoms,
        bonds=bonds,
        coords=coords.tolist(),
        canvas_size=CANVAS_SIZE,
        hl_atoms=hl_atoms,
    )


def remove_bonds(emol, atom):
    "Remove all of the bonds involving `atom` in place (?)"
    for bond in emol.GetMol().GetBonds():
        b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom in [b1, b2]:
            emol.RemoveBond(b1, b2)


def is_radical(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


@app.route("/update-molecule", methods=["POST"])
def update_molecule():
    data = request.get_json()
    global CUR_EDIT_MOL
    atoms, pid = data["atoms"], data["pid"]
    # kekulize before removing, sanitize after (openff_clean). from:
    # https://sourceforge.net/p/rdkit/mailman/message/37610064/
    Chem.Kekulize(CUR_EDIT_MOL, clearAromaticFlags=True)
    was_radical = is_radical(CUR_EDIT_MOL)
    emol = Chem.EditableMol(CUR_EDIT_MOL)
    for atom in sorted(atoms, reverse=True):
        remove_bonds(emol, atom)
        emol.RemoveAtom(atom)

    mol = openff_clean(emol.GetMol())
    if is_radical(mol) and not was_radical:
        for atom in mol.GetAtoms():
            while _n := atom.GetNumRadicalElectrons():
                atom.SetNumRadicalElectrons(_n - 1)
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
        mol = Chem.AddHs(mol)

    is_rad = is_radical(mol)

    # okay if it was already a radical, but if not, it shouldn't be now either
    assert (
        was_radical or not is_rad
    ), f"was radical? {was_radical}, is_radical? {is_rad}"

    logger.debug(f"updated mol: {mol_to_smiles(mol)}")
    mol, ret = mol_to_js(mol, pid)
    CUR_EDIT_MOL = mol
    return ret, HTTPStatus.CREATED


@app.route("/add-edit-mol", methods=["POST"])
def add_edit_mol():
    global CUR_EDIT_MOL
    smiles = mol_to_smiles(CUR_EDIT_MOL, mapped=True)
    data = request.get_json()
    pid, hl_atoms = data["pid"], data["hl_atoms"]
    logger.debug(
        f"add_edit_mol: smiles = {smiles}, pid = {pid}, atoms = {hl_atoms}"
    )
    table = Store.quick()
    table.add_to_dataset(smiles, pid, hl_atoms)
    return "", HTTPStatus.CREATED


@app.route("/preview-dataset")
def preview_dataset():
    table = Store.quick()
    draw_mols = []
    pids = []
    ids = []
    for id, s, pid, hl_atoms in table.get_dataset_entries():
        mol = mol_from_mapped_smiles(s)
        draw_mols.append(
            mol_to_draw(mol, pid, mol.GetNumAtoms(), atoms=hl_atoms)
        )
        pids.append(pid)
        ids.append(id)
    template = env.get_template("preview.html")
    return template.render(mols=draw_mols, pids=pids, ids=ids)


@app.route("/export-dataset", methods=["POST"])
def export_dataset():
    body = request.get_data(as_text=True)
    [label, filename] = body.split("=")
    assert label == "filename"
    table = Store.quick()
    with open(filename, "w") as out:
        for _id, smiles, pid, hl_atoms in table.get_dataset_entries():
            print(pid, smiles, hl_atoms, file=out)
    return redirect(url_for("index"))


@app.route("/reset-dataset", methods=["POST"])
def reset_dataset():
    table = Store.quick()
    table.reset_dataset()
    return redirect(url_for("index"))
