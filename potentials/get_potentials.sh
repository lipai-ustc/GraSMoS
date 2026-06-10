# lammps potentials
svn export https://github.com/lammps/lammps/branches/develop/potentials
mv potentials lammps

# deepmd potentials
mkdir deepmd
cd deepmd
wget https://github.com/lipai-ustc/Au-111-herringbone-reconstruction/blob/main/Au-PBE-novdw.pb
wget https://github.com/lipai-ustc/Theoretical-Study-on-the-Thermodynamics-of-Si-001-Surface/blob/main/Si(100)-SCAN.pb
wget https://zenodo.org/records/10215578/files/deepcnt-22.pb
cd ..

# pretrained models

