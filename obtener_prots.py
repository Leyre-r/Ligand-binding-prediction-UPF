import requests
import random

# A) PDB IDs con ligando
search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
query = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.nonpolymer_entity_count",
            "operator": "greater",
            "value": 0
        }
    },
    "return_type": "entry",
    "request_options": {"return_all_hits": True}
}

resp = requests.post(search_url, json=query, timeout=120)
data = resp.json()
pdbs_con_ligando = {x["identifier"].upper() for x in data["result_set"]}

# B) Clusters 30%
clusters_url = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt"
clusters_text = requests.get(clusters_url, timeout=120).text

seleccionados = []

for line in clusters_text.splitlines():
    members = line.strip().split()
    if not members:
        continue

    pdbs_cluster = []
    for member in members:
        pdb_id = member.split("_")[0].upper()
        if pdb_id in pdbs_con_ligando:
            pdbs_cluster.append(pdb_id)

    pdbs_cluster = list(dict.fromkeys(pdbs_cluster))
    if pdbs_cluster:
        seleccionados.append(random.choice(pdbs_cluster))

seleccionados = list(dict.fromkeys(seleccionados))
subset_70 = seleccionados[:70]

import pandas as pd

df = pd.DataFrame(subset_70, columns=["pdb_id"])
df.to_csv("pdb_ids_filtrados.csv", index=False)

print("IDs guardados en pdb_ids_filtrados.csv")

