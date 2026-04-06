# Ligand binding prediction
This repository contains the **Ligand Binding Prediction** project, a Machine Learning pipeline designed to identify protein binding sites using geometric and physicochemical surface descriptors. 

Developed as part of *Introduction to Python* and *Structural Bioinformatics* courses (MSc Bioinformatics for Health Sciences, UPF/UB).

## Tutorial

We recommend **creating a virtual environment** for the installation of the project:
    python -m venv binding_prediction
    source binding_prediction/bin/activate

**Install the dependencies**: the project can be directly installed from the distributed source file, which is in the dist/ directory:
    pip install dist/ligand_binding_prediction_upf-0.1.0.tar.gz

**Execution**: Once installed, you can run the program from any directory (e.g., from a folder containing your PDB files), provided the virtual environment is active:
    python -m python_project.inference path/your_protein.pdb

**Outputs:** The program generates two files:

    *_binding_site_residues.txt: A list of predicted amino acids.

    *_visualization.pml: A script to visualize the predicted pocket in PyMOL.

## Analysis of Examples
To replicate the results analysis presented, please open the Jupyter Notebook located in the tests/ directory: tests/binding_site_evaluation.ipynb.
    

## Theory 
### Introduction

Proteins bind to many types of molecules using a wide variety of binding sites. The current accepted model to explain the binding between a protein and a ligand is the “Conformational Selection Model”, in which the protein may adopt different conformations in its unbound state and a ligand binds selectively to one of these pre-existing conformations. This model has been supported by numerous experiments for both allosteric and nonallosteric ligands (Stank et al., 2016).

Understanding molecular interactions is useful for the study of natural and disease states, but also for the design of compounds with application in the pharmaceutical and biotechnological domains (Henrich et al., 2010). For that matter, ligand binding prediction is useful, because it avoids the step of having the protein crystalized or an NMR analysis to determine where the ligand binds.

### The Physico-Chemical Nature of Binding Sites

The binding pocket is usually defined as a cavity on the surface or in the interior of a protein that possesses suitable properties for binding a ligand. The set of amino acids around the binding pocket determines its physicochemical characteristics and, together with its shape and location in a protein, defines its functionality. Therefore, the main properties for characterizing a protein binding pocket are the overall geometry, the composition of amino acid residues, the type of solvation, the hydrophobicity, the electrostatics, and the chemical fragment interactions (Henrich et al., 2010).

### Feature Descriptors
To predict the ligandability of a certain point of the protein structure, this project takes into account the following binding pocket properties, using as a reference the top features described for P2Rank, a machine learning based tool that predicts ligand binding sites from the protein structure (Krivák & Hoksza, 2018).

1. **Protrusion**: The identification of pockets and cavities is relevant, since the binding sites for small molecules are usually pockets or crevices on the protein surface or cavities in the protein (Henrich et al., 2010).
We have defined the protrusion descriptor as the number of protein atoms within a sphere of 10 Å around a Solvent-Accessible Surface (SAS) point. This descriptor indicates the point’s “buriedness”. When its value is high, the SAS point belongs to a pocket and when its value is low, the SAS point is in a protrusion.

2. **b-factor**: It’s the weighted average, normalized by density, of the B-factor of the atoms that are within a radius of  6 Å from the SAS point. This feature is relevant, because binding sites often have a specific flexibility to be able to bind to the ligand. 

3. **Invalids**: Counts the number of Nitrogen (N) and Oxigen (O) atoms present within a 6 Å radius from the SAS point, weighting their contribution by their distance to the point. The invalids descriptor identifies regions with high polar density, which is favourable for the binding as Bartlett et al. (2002)
 found that catalytic residues occur with a higher frequency as charged residues than as polar or hydrophobic residues. 

4. **Aromatic**: Identifies the number of PHE, TYR and TRP residues present in a 6 Å radius. This is relevant because the occurrence of tryptophan was much higher at ligand binding sites, as proposed by Soga et al. (2007).

5. **Hydrophobic**: for each SAS point, it calculates the average hydrophobicity of the atoms within a 6 Å radius, weighted by the inverse of the distance and normalized by the local atomic density. The hydrophobicity is calculated using the scale Kyte-Doolittle (Kyte & Doolittle, 1982). This parameter allows to predict exposed or buried regions. 

6. **Polarity**: this parameter represents the average local polarity at a SAS point. It is calculated as the distance-weighted sum of the polarity values of neighboring amino acid side chains at physiological pH, normalized by the local atomic density.

7. **Net charge**: indicates the average weighted electrical charge of a point, computed using the charge values of the side chains associated with each atom found within a 6 Å radius. These values correspond to the standard charge of the amino acid residues at physiological pH (Pace et al., 2009).

8. **Ratio density**: measures the curvature or depth of the SAS point, calculated as the number of atoms found in a radius of 6.0 Å divided by the number of atoms found in a radius of 10.0 Å plus one.

9. **bfactor variance**: it’s the variance of the b-factor and measures the heterogeneity of the local flexibility. 

10. **hydro polar ratio**: defines the predominant chemical nature of the environment. It’s calculated as the division between the sum of the hydrophobicty and the polarity of the neighbouring atoms. 

11. **Unique residues**: counts the number of unique amino acids found within a radius of 6.0 Å, measuring the biochemical complexity of the environment that surrounds the SAS point.

### Computational Approach

The difficulty lies in developing procedures that are generally applicable for the identification of binding pockets, across all protein binding sites, as these sites vary in the relative importance of the different interactions contributing to binding (Henrich et al., 2010). Precisely, since it is unknown which interaction will be the most important for a specific protein, we have used a Random Forest Model. This model allows us to capture non-linear relationships between descriptors, without assuming that one descriptor is more important than another.

To address the challenge of developing more generalizable procedures, we have chosen a grid-based method: discretizing the protein into a 3D grid and analyzing the environment of each grid point found in the Solvent Accessible Surface (SAS). For our approach, we generated a grid enclosing the protein and selected the SAS points by filtering those located at a distance between 2.8 and 3.5 Å from the nearest atom.

Next, we calculated feature descriptors for those SAS points, taking into account the properties of the protein atoms surrounding them at distances of 6.0 and 10.0 Ångströms. Each atom's contribution was weighted based on its distance to the SAS point. The feature descriptors selected are an approximation of the top features described by P2Rank (Krivák & Hoksza, 2018).

This process was followed by a prediction of the ligandability score of each SAS point using a previously trained Random Forest model. While a standard approach would classify points using a fixed probability threshold (e.g., 0.5), this strategy does not account for the spatial nature of binding sites, which are known to form coherent surface regions rather than isolated points (Guilloux et al., 2009; Weisel et al., 2007).

To address this limitation, we implemented a spatial post-processing strategy that incorporates neighborhood information into the predictions.

First, a spatial smoothing step is applied. For each SAS point, its probability is replaced by the average probability of its neighboring points within a 4.0 Å radius. This reduces isolated noisy predictions and promotes spatial consistency.

Next, a local density measure is computed for each point by counting the number of neighboring SAS points within a 6.0 Å radius. This captures the idea that true binding sites tend to form dense clusters rather than isolated points (Guilloux et al., 2009).

These two signals are then combined into a final score:
- 80% weight: smoothed probability (model confidence)
- 20% weight: normalized local density (spatial support)

Finally, instead of using a fixed probability threshold, we apply an adaptive percentile-based threshold. For each protein, only the top-scoring SAS points (above the 94th percentile) are selected as binding-site candidates.

This approach allows the method to adapt to different proteins and improves robustness by prioritizing spatially coherent regions over isolated high-confidence predictions.

### Bibliography

Bartlett, G. J., Porter, C. T., Borkakoti, N., & Thornton, J. M. (2002). Analysis of catalytic residues in enzyme active sites. Journal of molecular biology, 324(1), 105–121. https://doi.org/10.1016/s0022-2836(02)01036-7

Henrich, S., Salo-Ahen, O.M.H., Huang, B., Rippmann, F.F., Cruciani, G. and Wade, R.C. (2010), Computational approaches to identifying and characterizing protein binding sites for ligand design. J. Mol. Recognit., 23: 209-219. https://doi.org/10.1002/jmr.984 

Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. Journal of molecular biology, 157(1), 105–132. https://doi.org/10.1016/0022-2836(82)90515-0

Krivák, R., & Hoksza, D. (2018). P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. Journal of cheminformatics, 10(1), 39. https://doi.org/10.1186/s13321-018-0285-8 

Pace, C. N., Grimsley, G. R., & Scholtz, J. M. (2009). Protein ionizable groups: pK values and their contribution to protein stability and solubility. Journal of Biological Chemistry, 284(20), 13289-13293.

Soga S, Shirai H, Kobori M, Hirayama N. 2007. Use of amino acid composition to predict ligand-binding sites. J. Chem. Inf. Model. 47: 400–406. https://doi.org/10.1021/ci6002202 

Stank, A., Kokh, D. B., Fuller, J. C., & Wade, R. C. (2016). Protein binding pocket dynamics. Accounts of Chemical Research, 49(5), 809–815. https://pubs.acs.org/doi/10.1021/acs.accounts.5b00516 

Guilloux, V. L., Schmidtke, P., & Tuffery, P. (2009). Fpocket: An open source platform for ligand pocket detection. BMC Bioinformatics, 10(1), 168. https://doi.org/10.1186/1471-2105-10-168

Weisel, M., Proschak, E., & Schneider, G. (2007). PocketPicker: analysis of ligand binding-sites with shape descriptors. Chemistry Central Journal, 1(1), 7. https://doi.org/10.1186/1752-153x-1-7





