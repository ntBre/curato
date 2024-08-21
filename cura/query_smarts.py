# conceptually identical to query but loads and searches for a a sequence of
# SMIRKS: name pairs instead of a sequence of parameter IDs tied to a force
# field

import itertools
import logging
from functools import partial
from multiprocessing import Pool

import click
from rdkit import Chem
from tqdm import tqdm

from cura.query import inner, parse_filters
from cura.store import DBForceField, Match, Store

logger = logging.getLogger(__name__)


def load_want(filename) -> list[tuple[str, Chem.Mol]]:
    params = []
    name_to_smirks = dict()
    with open(filename) as f:
        for line in f:
            [smarts, name] = line.split()
            params.append((name, Chem.MolFromSmarts(smarts)))
            name_to_smirks[name] = smarts
    return params, name_to_smirks


def _main(
    nprocs,
    chunk_size,
    filters,
    store_name,
    ffname,
    params,
    limit,
    pid_to_smirks,
):
    s = Store(store_name)

    s.reset_forcefield(ffname)

    res = dict()
    mols = [(s, False) for s in s.get_molecules(limit)]
    frags = [(s, True) for s in s.get_fragments(limit)]
    all_mols = [i for i in itertools.chain(mols, frags)]
    unmatched = 0
    with Pool(processes=nprocs) as p:
        for smiles, matches, is_frag in tqdm(
            p.imap(
                partial(inner, params=params, filters=filters),
                all_mols,
                chunksize=chunk_size,
            ),
            total=len(all_mols),
        ):
            for pid in matches:
                if pid not in res:
                    res[pid] = Match(pid_to_smirks[pid], pid, list(), list())
                if is_frag:
                    res[pid].fragments.append(smiles)
                else:
                    res[pid].molecules.append(smiles)
            else:
                unmatched += 1

    logger.warning(f"{unmatched} SMILES not matching desired parameters")

    for pid, mat in res.items():
        print(f"{pid} {len(mat.fragments)} frags {len(mat.molecules)} mols")

    ret = list(res.values())
    s.insert_forcefield(DBForceField(ffname, ret))

    if logger.level > logging.WARNING:
        for mat in ret:
            print(mat.pid, len(mat.molecules))

    return s, ret


@click.command()
@click.option("--nprocs", "-n", default=8)
@click.option("--chunk-size", "-c", default=32)
@click.option("--filter", "-x", "filters", multiple=True)
@click.option("--store-name", "-s", default="store.sqlite")
@click.option("--target-params", "-t", default="want.params")
@click.option("--limit", "-l", hidden=True, default=None)
def main(nprocs, chunk_size, filters, store_name, target_params, limit):
    filters = parse_filters(filters)
    want, pid_to_smirks = load_want(target_params)
    _main(
        nprocs,
        chunk_size,
        filters,
        store_name,
        target_params,
        want,
        limit,
        pid_to_smirks,
    )


if __name__ == "__main__":
    main()
