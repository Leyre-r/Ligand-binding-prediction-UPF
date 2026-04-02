# Ligand binding prediction
This repository contains the Ligand Binding Prediction project developed during our Introduction to Python and Structural Bioinformatics course (MSc Bioinformatics for Health Sciences, UPF/UB).

## Tutorial

Install dependencies: pip install -r requirements.txt

Prepare the PDB file: "Coloca tu archivo .pdb en la carpeta del programa"

Execute the program: python inferencia2.py --input protein.pdb

## Theory 
### Introduction

Proteins bind to many types of molecules using a wide variety of binding sites. The current accepted model to explain the binding between a protein and a ligand is the “Conformational Selection Model”, in which the protein may adopt different conformations in its unbound state and a ligand binds selectively to one of these pre-existing conformations. This model has been supported by numerous experiments for both allosteric and nonallosteric ligands (Stank et al., 2016).

Understanding molecular interactions is useful for the study of natural and disease states, but also for the design of compounds with application in the pharmaceutical and biotechnological domains (Henrich et al., 2010). For that matter, ligand binding prediction is useful, because it avoids the step of having the protein crystalized or an NMR analysis to determine where the ligand binds.

### The Physico-Chemical Nature of Binding Sites

The binding pocket is usually defined as a cavity on the surface or in the interior of a protein that possesses suitable properties for binding a ligand. The set of amino acids around the binding pocket determines its physicochemical characteristics and, together with its shape and location in a protein, defines its functionality. Therefore, the main properties for characterizing a protein binding pocket are the overall geometry, the composition of amino acid residues, the type of solvation, the hydrophobicity, the electrostatics, and the chemical fragment interactions (Henrich et al., 2010).

### Feature Descriptors
To predict the ligandability of a certain point of the protein structure, this project takes into account the following binding pocket properties, using as a reference the top features described for P2Rank, a machine learning based tool that predicts ligand binding sites from the protein structure (Krivák & Hoksza, 2018).


### Computational Approach
