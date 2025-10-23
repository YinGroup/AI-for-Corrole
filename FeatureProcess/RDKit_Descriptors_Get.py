import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_descriptors(mol):
    """
    Given an RDKit Mol object, calculate all built-in RDKit descriptors for the molecule.
    Returns a list or dictionary structure.
    """
    desc_funcs = [desc_tuple[1] for desc_tuple in Descriptors._descList]

    desc_values = []
    for func in desc_funcs:
        try:
            value = func(mol)
            desc_values.append(value)
        except Exception as e:
            # If a descriptor calculation fails, return NaN to prevent program interruption
            print(f"Warning: Descriptor {func.__name__} calculation failed: {e}")
            desc_values.append(float('nan'))
    return desc_values


def main():
    # List of RDKit built-in descriptor names
    desc_names = [desc_tuple[0] for desc_tuple in Descriptors._descList]

    # Build DataFrame to store descriptor results for all molecules
    columns = ["Mol_ID"] + desc_names
    df_descriptors = pd.DataFrame(columns=columns)

    # Get all files in the current folder
    file_list = os.listdir('.')
    mol2_files = [f for f in file_list if f.endswith('.mol2')]

    if not mol2_files:
        print("No .mol2 files found in the current folder, program terminated.")
        return

    print(f"Found {len(mol2_files)} .mol2 files, starting descriptor calculation...")

    for mol2_file in mol2_files:
        mol_id = os.path.splitext(mol2_file)[0]  # Use filename as ID, without extension

        try:
            mol = Chem.MolFromMol2File(mol2_file, sanitize=True)
            if mol is None:
                print(f"Warning: {mol2_file} parsing failed, skipping.")
                continue
        except Exception as e:
            print(f"Error: Reading {mol2_file} failed: {e}")
            continue

        # Calculate molecular descriptors
        desc_values = calculate_descriptors(mol)

        # Append the result to the DataFrame
        row_data = [mol_id] + desc_values
        df_descriptors.loc[len(df_descriptors)] = row_data

    # Save the results to Excel
    output_excel = "molecular_descriptors.xlsx"
    df_descriptors.to_excel(output_excel, index=False)
    print(f"Descriptor calculation complete, saved to {output_excel}.")


if __name__ == "__main__":
    main()