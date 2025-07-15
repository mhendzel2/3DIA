# PyMOL Script for Visualizing Lamin A/C Ig-Fold Mutations by Phenotype
#
# This script fetches the PDB structure 1IFR, displays its molecular surface,
# and colors key mutation sites according to their associated disease category.
#
# Color Legend:
# - Myopathy/Cardiomyopathy (Red): Residues where mutations primarily cause
#   muscular dystrophies (like EDMD) or cardiomyopathies.
# - Lipodystrophy (Blue): Residues where mutations primarily cause metabolic
#   disorders like FPLD2.
# - Progeroid Syndromes (Orange): Residues where mutations primarily cause
#   premature aging syndromes like MADA or AHGPS.
# - Overlap Phenotypes (Magenta): Residues where different mutations at the
#   same site cause distinct major phenotypes (e.g., myopathy and progeria).

from pymol import cmd

# --- 1. Configuration ---

# PDB ID for the Lamin A/C Ig-like domain
pdb_id = '1ifr'

# Define a color scheme for each phenotype category
# Using PyMOL's named colors for clarity
color_myopathy = 'firebrick'
color_lipodystrophy = 'marine'
color_progeroid = 'orange'
color_overlap = 'magenta'

# Define lists of residues for each category based on published data.
# Residue numbers correspond to UniProt P02545, which match the PDB file.
residues_myopathy = ['445', '453', '481', '514']
residues_lipodystrophy = ['482', '486']
residues_progeroid = ['471']
residues_overlap = ['527'] # R527P causes myopathy; R527H causes progeria

# --- 2. Visualization Setup ---

# Fetch the PDB structure from the RCSB database
cmd.fetch(pdb_id, async_=0)

# Set a clean, white background for better contrast
cmd.bg_color('white')

# Remove water molecules and other non-protein entities
cmd.remove('solvent')

# Display the protein as a solid surface
cmd.show('surface')

# Apply a neutral base color to the entire protein surface
cmd.color('grey80', pdb_id)

# --- 3. Create Selections and Apply Colors ---

# Create a selection for each group of residues
# The 'resi' selector uses residue numbers. We join the list with '+'
# to select multiple residues in one command.
cmd.select('myopathy_sites', 'resi ' + '+'.join(residues_myopathy))
cmd.select('lipodystrophy_sites', 'resi ' + '+'.join(residues_lipodystrophy))
cmd.select('progeroid_sites', 'resi ' + '+'.join(residues_progeroid))
cmd.select('overlap_sites', 'resi ' + '+'.join(residues_overlap))

# Apply the defined colors to the surface of each selection
cmd.color(color_myopathy, 'myopathy_sites')
cmd.color(color_lipodystrophy, 'lipodystrophy_sites')
cmd.color(color_progeroid, 'progeroid_sites')
cmd.color(color_overlap, 'overlap_sites')

# --- 4. Final Touches ---

# Center the view on the protein and orient it
cmd.zoom(pdb_id)
cmd.orient()

# Print a summary of actions to the PyMOL console for the user
print("--- Visualization Complete: Lamin A/C Ig-Fold Mutations ---")
print(f"Loaded and processed PDB structure: {pdb_id}")
print("Residues have been highlighted on the surface based on disease phenotype:")
print(f"  - Myopathy/Cardiomyopathy sites colored {color_myopathy}: {', '.join(residues_myopathy)}")
print(f"  - Lipodystrophy sites colored {color_lipodystrophy}: {', '.join(residues_lipodystrophy)}")
print(f"  - Progeroid sites colored {color_progeroid}: {', '.join(residues_progeroid)}")
print(f"  - Overlap sites colored {color_overlap}: {', '.join(residues_overlap)}")
print("-------------------------------------------------------------")

