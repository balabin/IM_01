#!/bin/env/python
# Circular fingerprints and JVM start
# Ilya Balabin <ibalabin@avicenna-bio.com>

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors

# generate fingerprint of choice
def FingerprintsFromSmiles(smiles, name, size):
    if name=='morgan3':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=size)
        return [fpgen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
    else:
        return None