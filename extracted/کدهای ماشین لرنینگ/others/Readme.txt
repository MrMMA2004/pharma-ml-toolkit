adder:
adds files or anything to folders that are in the main folder
	check the location and the file name

checker:
checks the folders that are in main folder to see if there is a r.dlg or not and completed or not

delete:
keeps the files you want and deletes others
	check file name and location

main:
automated docking code

reader:
reads the r.dlgs in all folders located in main folder and reports num. in cluster mean and the lowest binding energy

smiles_to_pdb:
converts list of smiles to pdb files by rdkit and anaconda promts
	check folder and file names

common between two excel:
gets common words between two excel files
	check path and column

approved:
gets approved ones from tsv of dgidb
	some of them are market approved not fda approved

sdf reading:
instructions to read and edit the sdfs
	open with jupyter-lab

gpfer:
produces r.gpf for all subfolders in the main folder
	check that receptor file should be r.pdbqt and you have to give the grid center and size, main folder and the receptor type manually in the code

gpfreader:
checks the ligand types of all r.gpfs in all subfolders and gives the data
	use it to check the accuracy of the gpfer.py code

deleter:
deletes the file you want in subfolders
	check the file name and path

dpfer:
produces r.dpf for all subfolders in the main folder

smiqt:
produces pdbqts of ligands from 'SMILES' in excel file

smilestopdbqt:
gives pdbqt from smiles
	check that you have a column named SMILES in ligands.csv
	it's edited and ready for autodock4.2
docker:
docks parallely 
	edited form of main

gpferreceptorni:
used for the receptors that have ni, zn, ...
	check receptor type and gridbox cordinations
	r.dpf and main.py is the same if the problem is just the receptor

extaacter:
extracts the r.dlgs that are in subfolders of a main folder and names it the folder name

readerwithdetails:
gives the informations about the best cluster and the most crowded cluster.
	gives the lowest/mean binding energy and the no. in cluster.

nameextracter:
extracts the names of all csvs in a main folder and write it in a csv file