"""
inference.py
------------------------
Inference module for protein binding-site prediction.

This script predicts binding sites from a PDB file using a trained Random Forest model. The pipeline:

1. Computes SAS-based descriptors (same as training).
2. Applies the trained classifier to each SAS point.
3. Refines probabilities using spatial smoothing and local density.
4. Maps predicted surface points to protein residues.
5. Exports results for analysis and visualization.

Usage:
    python inference.py <protein.pdb>

Outputs:
    - <pdb_name>_binding_site_residues.txt: Predicted binding-site residues.
    - <pdb_name>_visualization.pml: PyMOL visualization script.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from python_project.grid import PROPERTIES



# LOGGER CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "inference.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_features(pdb_file):
    """
    Compute SAS-based structural and physicochemical descriptors from a PDB file.

    This function reproduces the feature engineering pipeline used during model training. 
    It parses the structure, filters standard protein atoms (excluding HETATM entries), generates a 3D grid, 
    and selects approximate solvent-accessible surface (SAS) points based on distance criteria.

    For each SAS point, local atomic neighborhoods are analyzed to compute descriptors such as hydrophobicity, 
    aromaticity, polarity, charge, B-factor-derived flexibility, density, and residue diversity.

    Args:
        pdb_file (str): Path to the input PDB file.

    Returns:
        tuple[pd.DataFrame, np.ndarray, list, object, KDTree] | tuple[None, None, None, None, None]:
            - df: Feature table (one row per SAS point).
            - sas_points: Coordinates of SAS points.
            - atoms: Filtered Bio.PDB.Atom objects.
            - structure: Parsed structure object.
            - tree: KDTree built on atom coordinates.
            Returns (None, None, None, None, None) if an error occurs.

    Note:
        Feature computation must remain identical to training. Any change in atom selection, thresholds, 
        or physicochemical properties may invalidate model predictions.
    """

    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        return None, None, None, None, None

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("protein", pdb_file)
    except Exception as e:
        logger.error(f"Error parsing PDB file '{pdb_file}': {e}")
        return None, None, None, None, None

    # Keep only standard protein atoms. This mirrors the original logic and excludes ligands, waters, and other HETATM-derived entities.
    atoms = [
        atom for atom in structure.get_atoms()
        if atom.get_parent().get_id()[0] == ' '
    ]

    if len(atoms) == 0:
        logger.error("No standard protein atoms found in the structure.")
        return None, None, None, None, None

    coords = np.array([atom.get_coord() for atom in atoms], dtype=float)
    tree = KDTree(coords)

    try:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    except Exception as e:
        logger.error(f"Error computing structure bounding box: {e}")
        return None, None, None, None, None

    # Grid-based SAS approximation.
    # Keep grid points located at an intermediate distance from the protein surface, roughly representing solvent-accessible points.
    grid_points = np.mgrid[
        x_min:x_max:1,
        y_min:y_max:1,
        z_min:z_max:1
    ].reshape(3, -1).T

    distances, _ = tree.query(grid_points)
    mask = (distances > 2.8) & (distances < 3.5)
    sas_points = grid_points[mask]

    if len(sas_points) == 0:
        logger.error("No SAS points were generated from the input structure.")
        return None, None, None, None, None

    logger.info(f"SAS points generated: {len(sas_points)}")

    data_rows = []

    for point in sas_points:
        neighbors_6A = tree.query_ball_point(point, 6.0)
        neighbors_10A = tree.query_ball_point(point, 10.0)

        density_6A = len(neighbors_6A)

        f_hydro = 0.0
        f_aromatic = 0.0
        f_polar = 0.0
        f_charge = 0.0
        f_bfactor = 0.0
        f_invalids = 0.0

        for idx in neighbors_6A:
            atom = atoms[idx]
            residue_name = atom.get_parent().get_resname()

            distance = np.linalg.norm(point - atom.get_coord())
            weight = 1.0 / (distance + 0.5)

            if residue_name in PROPERTIES:
                props = PROPERTIES[residue_name]

                f_hydro += props['hydro'] * weight
                f_aromatic += props['aromatic'] * weight
                f_polar += props['polar'] * weight
                f_charge += props['charge'] * weight
                f_bfactor += atom.get_bfactor() * weight

                atom_name = atom.get_name().strip()
                if atom_name.startswith(('N', 'O')):
                    f_invalids += weight

        # Normalization by local density helps reduce bias toward crowded regions.
        hydro_norm = f_hydro / (density_6A + 1)
        polar_norm = f_polar / (density_6A + 1)
        charge_norm = f_charge / (density_6A + 1)
        bfactor_norm = f_bfactor / (density_6A + 1)

        ratio_density = len(neighbors_6A) / (len(neighbors_10A) + 1)
        hydro_polar_ratio = f_hydro / (f_polar + 1)

        unique_residues = len(set(
            atoms[idx].get_parent().get_resname()
            for idx in neighbors_6A
        ))

        if density_6A > 0:
            bfactor_var = np.var([
                atoms[idx].get_bfactor()
                for idx in neighbors_6A
            ])
        else:
            bfactor_var = 0.0

        data_rows.append({
            'protrusion': len(neighbors_10A),
            'bfactor': bfactor_norm,
            'Invalids': f_invalids,
            'Aromatic': f_aromatic,
            'hydrophobic': hydro_norm,
            'polar': polar_norm,
            'net_charge': charge_norm,
            'ratio_density': ratio_density,
            'bfactor_var': bfactor_var,
            'hydro_polar_ratio': hydro_polar_ratio,
            'unique_residues': unique_residues
        })

    df = pd.DataFrame(data_rows)

    if df.empty:
        logger.error("Feature DataFrame is empty after computation.")
        return None, None, None, None, None

    return df, sas_points, atoms, structure, tree


def map_residues(sas_points, predictions, atoms, tree, radius=5.0):
    
    """
    Map predicted binding-site SAS points to protein residues.

    For each SAS point classified as positive, nearby atoms within a given radius are retrieved and their 
    parent residues are collected. The result is a unique, sorted list of residues.

    Args:
        sas_points (np.ndarray): Coordinates of SAS points.
        predictions (np.ndarray): Binary array (1 = binding site, 0 = non-binding).
        atoms (list): Bio.PDB.Atom objects corresponding to the KDTree.
        tree (KDTree): KDTree built from atom coordinates.
        radius (float, optional): Neighborhood radius in angstroms. Defaults to 5.0.

    Returns:
        list[tuple[str, int, str]]: Sorted list of unique residues as
            (chain_id, residue_number, residue_name).
            Returns an empty list if no positive predictions are found.
    """
    if predictions is None or len(predictions) == 0:
        logger.warning("Prediction array is empty.")
        return []

    if np.sum(predictions) == 0:
        logger.warning("No binding-site points were predicted.")
        return []

    binding_points = sas_points[predictions == 1]
    residues = set()

    neighbor_indices = tree.query_ball_point(binding_points, radius)

    for neighbors in neighbor_indices:
        for idx in neighbors:
            atom = atoms[idx]
            residue = atom.get_parent()
            chain_id = residue.get_parent().get_id()
            residue_number = residue.get_id()[1]
            residue_name = residue.get_resname()

            residues.add((chain_id, residue_number, residue_name))

    return sorted(residues, key=lambda x: (x[0], x[1]))


def save_residues_txt(residues, pdb_file, output="binding_site_residues.txt"):
    """
    Save predicted binding-site residues to a text file.

    The output includes the PDB filename, total number of residues, and a formatted table with chain ID, 
    residue number, and residue name.

    Args:
        residues (list[tuple[str, int, str]]): Predicted residues as (chain_id, residue_number, residue_name).
        pdb_file (str): Path to the input PDB file.
        output (str, optional): Output file path. Defaults to "binding_site_residues.txt".

    Returns:
        bool: True if the file is written successfully, False otherwise.
    """
    try:
        pdb_name = os.path.basename(pdb_file)

        with open(output, "w", encoding="utf-8") as f:
            f.write(f"Binding site residues — {pdb_name}\n")
            f.write(f"Total residues: {len(residues)}\n")
            f.write("=" * 40 + "\n")
            f.write(f"{'Chain':<8}{'ResNum':<10}{'ResName'}\n")
            f.write("-" * 40 + "\n")

            for chain, residue_number, residue_name in residues:
                f.write(f"{chain:<8}{residue_number:<10}{residue_name}\n")

        logger.info(f"Residue list saved to: {output}")
        return True

    except Exception as e:
        logger.error(f"Error writing residue report '{output}': {e}")
        return False


def save_pymol_script(residues, pdb_file, output="visualization.pml"):
    """
    Generate a PyMOL script to visualize predicted binding-site residues.

    The script loads the structure, displays it as a cartoon, highlights the predicted residues as sticks, 
    and renders a semi-transparent surface.

    Args:
        residues (list[tuple[str, int, str]]): Predicted residues as (chain_id, residue_number, residue_name).
        pdb_file (str): Path to the input PDB file.
        output (str, optional): Output script path. Defaults to "visualization.pml".

    Returns:
        bool: True if the script is written successfully, False otherwise.
    """
    try:
        pdb_path = os.path.abspath(pdb_file)
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

        if len(residues) == 0:
            logger.warning("No residues available for PyMOL selection.")
            return False

        selections_by_chain = {}
        for chain, residue_number, _ in residues:
            selections_by_chain.setdefault(chain, []).append(str(residue_number))

        selection_parts = []
        for chain, residue_ids in selections_by_chain.items():
            residue_string = "+".join(residue_ids)
            selection_parts.append(f"(chain {chain} and resi {residue_string})")

        selection_string = " or ".join(selection_parts)

        with open(output, "w", encoding="utf-8") as f:
            f.write("# PyMOL script — Binding site prediction\n")
            f.write("# Generated by inference.py\n\n")
            f.write(f"load {pdb_path}, {pdb_name}\n\n")
            f.write("# Base representation\n")
            f.write(f"hide everything, {pdb_name}\n")
            f.write(f"show cartoon, {pdb_name}\n")
            f.write(f"color gray80, {pdb_name}\n\n")
            f.write("# Binding-site selection\n")
            f.write(f"select binding_site, {pdb_name} and ({selection_string})\n\n")
            f.write("# Binding-site display\n")
            f.write("show sticks, binding_site\n")
            f.write("color red, binding_site\n")
            f.write("set stick_radius, 0.2\n\n")
            f.write("# Binding-site surface\n")
            f.write("create bs_surface, binding_site\n")
            f.write("show surface, bs_surface\n")
            f.write("color tv_red, bs_surface\n")
            f.write("set transparency, 0.4, bs_surface\n\n")
            f.write("# Final view\n")
            f.write("zoom binding_site\n")
            f.write("ray 1200, 900\n")

        logger.info(f"PyMOL script saved to: {output}")
        return True

    except Exception as e:
        logger.error(f"Error writing PyMOL script '{output}': {e}")
        return False


def predict_binding_site(pdb_file):
    """
    Run the full binding-site prediction pipeline.

    This function performs the complete inference workflow:
    1. Loads a trained Random Forest model.
    2. Computes SAS-based descriptors.
    3. Predicts class probabilities for each SAS point.
    4. Applies spatial smoothing.
    5. Computes local SAS-point density.
    6. Combines both signals into a final score.
    7. Applies an adaptive percentile-based threshold.
    8. Maps predicted points to protein residues.
    9. Exports results as a text report and a PyMOL script.

    Args:
        pdb_file (str): Path to the input PDB file.

    Returns:
        tuple[list[tuple[str, int, str]], np.ndarray, np.ndarray, np.ndarray] | None:
            - residues: Predicted binding-site residues.
            - predictions: Binary predictions for SAS points.
            - probabilities: Raw model probabilities.
            - score: Final combined score used for thresholding.
            Returns None if an error occurs.

    Note:
        The model is not modified. Improvements come from post-processing steps
        (smoothing, density weighting, and adaptive thresholding).
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "modelo_rf_predictor.pkl")

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model '{model_path}': {e}")
        return None

    result = compute_features(pdb_file)
    if result[0] is None:
        logger.error("Feature computation failed.")
        return None

    df_features, sas_points, atoms, _, atom_tree = result

    expected_columns = [
        'protrusion',
        'bfactor',
        'Invalids',
        'Aromatic',
        'hydrophobic',
        'polar',
        'net_charge',
        'ratio_density',
        'bfactor_var',
        'hydro_polar_ratio',
        'unique_residues'
    ]

    missing_columns = [col for col in expected_columns if col not in df_features.columns]
    if missing_columns:
        logger.error(f"Missing expected feature columns: {missing_columns}")
        return None

    X_pred = df_features[expected_columns]

    try:
        probabilities = model.predict_proba(X_pred)[:, 1]
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return None

    if len(probabilities) == 0:
        logger.error("Model returned an empty probability array.")
        return None

    # Build a KDTree on SAS points for inference refinement.
    # This is separate from the atom KDTree, which is used later for residue mapping.
    sas_tree = KDTree(sas_points)

    # Spatial smoothing:
    # each point gets the mean probability of its local neighborhood.
    # This reduces isolated spikes and makes predictions more spatially coherent.
    smooth_probabilities = []
    for point in sas_points:
        neighbors = sas_tree.query_ball_point(point, 4.0)
        if len(neighbors) == 0:
            smooth_probabilities.append(0.0)
        else:
            smooth_probabilities.append(np.mean(probabilities[neighbors]))
    smooth_probabilities = np.array(smooth_probabilities)

    # Local density:
    # denser regions of positive-like points are more likely to correspond
    # to meaningful pockets than isolated individual points.
    local_density = np.array([
        len(sas_tree.query_ball_point(point, 6.0))
        for point in sas_points
    ], dtype=float)

    if local_density.max() == 0:
        logger.warning("Local density is zero for all SAS points.")
        density_norm = local_density
    else:
        density_norm = local_density / local_density.max()

    # Final score:
    # we combine smoothed class confidence and spatial support.
    score = 0.8 * smooth_probabilities + 0.2 * density_norm

    if len(score) == 0:
        logger.error("Final score array is empty.")
        return None

    # Adaptive threshold:
    # the score cutoff is computed per protein from its own score distribution
    # using a fixed percentile threshold.

    PERCENTILE_THRESHOLD = 94
    threshold = np.percentile(score, PERCENTILE_THRESHOLD)
    predictions = (score >= threshold).astype(int)

    n_positive = int(predictions.sum())
    logger.info(f"Binding-site points selected: {n_positive} / {len(predictions)}")
    logger.info(f"Adaptive score threshold used: {threshold:.6f}")

    residues = map_residues(sas_points, predictions, atoms, atom_tree, radius=5.0)
    logger.info(f"Unique binding-site residues mapped: {len(residues)}")

    pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]

    save_residues_txt(
        residues,
        pdb_file,
        output=f"{pdb_base}_binding_site_residues.txt"
    )

    save_pymol_script(
        residues,
        pdb_file,
        output=f"{pdb_base}_visualization.pml"
    )

    logger.info("Prediction completed successfully.")

    return residues, predictions, probabilities, score


if __name__ == "__main__":
    """
    Command-line entry point.

    Usage:
        python inference.py <protein.pdb>
    """

    if len(sys.argv) < 2:
        logger.error("Usage: python inference.py <protein.pdb>")
        logger.error("Example: python inference.py 1OV9.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]

    try:
        predict_binding_site(input_pdb)
    except Exception as e:
        logger.exception(f"Unexpected error during execution: {e}")
        sys.exit(1)
