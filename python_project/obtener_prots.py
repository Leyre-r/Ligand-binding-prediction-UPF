import os
import pandas as pd

def encontrar_archivos_sample(ruta_sample):
    protein = None
    ligand = None
    pocket = None

    for f in os.listdir(ruta_sample):
        f_lower = f.lower()

        if "protein.pdb" in f_lower:
            protein = os.path.join(ruta_sample, f)

        elif "ligand.sdf" in f_lower:
            ligand = os.path.join(ruta_sample, f)

        elif "pocket.pdb" in f_lower:
            pocket = os.path.join(ruta_sample, f)

    return protein, ligand, pocket


def obtener_prots_definitivo(
    root_dir,
    output_csv="samples.csv",
    min_protein_size_kb=50,
    require_pocket=False,
    sample_size=None,
    seed=42
):
    """
    Recorre el dataset estructurado tipo PDBbind y genera un CSV limpio de samples.
    """

    import random
    random.seed(seed)

    samples = []
    total_folders = 0

    for era in os.listdir(root_dir):
        ruta_era = os.path.join(root_dir, era)

        if not os.path.isdir(ruta_era):
            continue

        print(f"Explorando carpeta: {era}")

        for pdb_id in os.listdir(ruta_era):
            ruta_sample = os.path.join(ruta_era, pdb_id)

            if not os.path.isdir(ruta_sample):
                continue

            total_folders += 1

            protein, ligand, pocket = encontrar_archivos_sample(ruta_sample)

            # ────────────────
            # FILTROS
            # ────────────────

            # 1. deben existir protein y ligand
            if protein is None or ligand is None:
                continue

            # 2. tamaño mínimo proteína
            if os.path.getsize(protein) < min_protein_size_kb * 1024:
                continue

            # 3. opcional: requerir pocket
            if require_pocket and pocket is None:
                continue

            samples.append({
                "pdb_id": pdb_id,
                "protein_path": protein,
                "ligand_path": ligand,
                "pocket_path": pocket if pocket else ""
            })

    print(f"\nTotal carpetas analizadas: {total_folders}")
    print(f"Samples válidos encontrados: {len(samples)}")

    # ────────────────
    # SAMPLING OPCIONAL
    # ────────────────
    if sample_size is not None and len(samples) > sample_size:
        samples = random.sample(samples, sample_size)
        print(f"Submuestreo aplicado: {len(samples)} samples")

    df = pd.DataFrame(samples)
    df = df.sort_values("pdb_id").reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"\nCSV guardado en: {output_csv}")

    return df


# ─────────────────────────────
# EJECUCIÓN
# ─────────────────────────────
if __name__ == "__main__":
    dataset_path = "/home/julia/Documentos/Segon Trimestre/PYT/Proyecto/P-L"  

    df = obtener_prots_definitivo(
        root_dir=dataset_path,
        output_csv="samples.csv",
        min_protein_size_kb=50,
        require_pocket=False,   # True si quieres usarlo más adelante
        sample_size=1000,       # pon 200 para pruebas rápidas
        seed=42
    )
