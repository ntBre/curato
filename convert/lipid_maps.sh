# Clean up the LIPID MAPS Structure Database (LMSD) TSV retrieved from
# https://www.lipidmaps.org/files/?file=lipidmaps_ids_cc0&ext=tsv. Assumes it
# was saved to lipidmaps.tsv

# skip header, then extract smiles field. fields should be ID, inchikey, smiles,
# and an optional field called obsolete_id
awk -F '\t' 'NR > 1 { print $3 }' lipidmaps.tsv >lipidmaps.smi
