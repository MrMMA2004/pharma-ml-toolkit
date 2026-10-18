import os

def extract_ligand_types(pdbqt_path):
    ligand_types = set()
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) >= 12:
                    atom_type = parts[-1].strip()
                    if atom_type.isalpha():
                        ligand_types.add(atom_type)
    return sorted(ligand_types)

def get_about_coords(pdbqt_path):
    x_list, y_list, z_list = [], [], []
    with open(pdbqt_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)
                except:
                    continue
    if not x_list:
        raise ValueError(f"❌ مختصات در {pdbqt_path} پیدا نشد.")
    cx = round(sum(x_list) / len(x_list), 3)
    cy = round(sum(y_list) / len(y_list), 3)
    cz = round(sum(z_list) / len(z_list), 3)
    return cx, cy, cz

def get_torsdof(pdbqt_path):
    with open(pdbqt_path, 'r') as file:
        for line in file:
            if line.startswith("TORSDOF"):
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    return int(parts[1])
    return 0

def generate_r_dpf(folder_path, receptor_file="r.pdbqt"):
    files = os.listdir(folder_path)

    # یافتن فایل لیگاند
    ligand_file = next((f for f in files if f.endswith(".pdbqt") and f != receptor_file), None)
    if not ligand_file:
        print(f"⚠️ لیگاند پیدا نشد در: {folder_path}")
        return

    ligand_path = os.path.join(folder_path, ligand_file)
    receptor_path = os.path.join(folder_path, receptor_file)

    try:
        ligand_types = extract_ligand_types(ligand_path)
        cx, cy, cz = get_about_coords(ligand_path)
        torsdof = get_torsdof(ligand_path)
    except Exception as e:
        print(f"[×] خطا در {folder_path}: {e}")
        return

    dpf_text = f"""autodock_parameter_version 4.2
outlev 1
intelec
seed pid time
ligand_types {' '.join(ligand_types)}
fld r.maps.fld
map """ + "\nmap ".join([f"r.{lt}.map" for lt in ligand_types]) + f"""
elecmap r.e.map
desolvmap r.d.map
move {ligand_file}
about {cx} {cy} {cz}
tran0 random
quaternion0 random
dihe0 random
torsdof {torsdof}
rmstol 2.0
extnrg 1000.0
e0max 0.0 10000
ga_pop_size 150
ga_num_evals 2500000
ga_num_generations 27000
ga_elitism 1
ga_mutation_rate 0.02
ga_crossover_rate 0.8
ga_window_size 10
ga_cauchy_alpha 0.0
ga_cauchy_beta 1.0
set_ga
sw_max_its 300
sw_max_succ 4
sw_max_fail 4
sw_rho 1.0
sw_lb_rho 0.01
ls_search_freq 0.06
set_psw1
unbound_model bound
ga_run 100
analysis
"""

    with open(os.path.join(folder_path, 'r.dpf'), 'w') as dpf_file:
        dpf_file.write(dpf_text)

    print(f"✅ ساخته شد: {folder_path}/r.dpf")

# اجرای کد در همه‌ی پوشه‌های داخل پوشه فعلی
def main():
    base = os.getcwd()
    for folder in os.listdir(base):
        path = os.path.join(base, folder)
        if os.path.isdir(path):
            generate_r_dpf(path)

if __name__ == "__main__":
    main()
