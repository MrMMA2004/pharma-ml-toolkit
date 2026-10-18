import os

# ============ 👇 تنظیمات دستی شما 👇 ============ #
receptor_types = "A C HD N Ni OA SA"  # ← اینو خودت تنظیم کن
grid_center = (127.863, 126.349, 87.545)
npts = (40, 40, 40)
spacing = 0.375
receptor_file = "r.pdbqt"  # فایل رسپتور ثابته
# =============================================== #

main_folder = os.getcwd()

for subfolder in os.listdir(main_folder):
    sub_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(sub_path):
        files = os.listdir(sub_path)

        # پیدا کردن فایل لیگاند
        ligand_file = None
        for f in files:
            if f.endswith(".pdbqt") and f != receptor_file:
                ligand_file = f
                break

        if not ligand_file:
            print(f"⚠️  لیگاند پیدا نشد در پوشه: {subfolder}")
            continue

        # استخراج atom types از لیگاند
        ligand_types = set()
        with open(os.path.join(sub_path, ligand_file), 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    parts = line.split()
                    if len(parts) > 2:
                        atom_type = parts[-1]
                        if atom_type.isalpha():
                            ligand_types.add(atom_type)

        ligand_types_str = " ".join(sorted(ligand_types))

        # نوشتن فایل r.gpf
        gpf_content = f"""parameter_file AD4_parameters.dat
npts {npts[0]} {npts[1]} {npts[2]}                        # num.grid points in xyz
gridfld r.maps.fld                   # grid_data_file
spacing {spacing}                        # spacing(A)
receptor_types {receptor_types}     # receptor atom types
ligand_types {ligand_types_str}            # ligand atom types
receptor {receptor_file}                     # macromolecule
gridcenter {grid_center[0]} {grid_center[1]} {grid_center[2]}      # xyz-coordinates
smooth 0.5                           # store minimum energy w/in rad(A)
"""

        for lt in sorted(ligand_types):
            gpf_content += f"map r.{lt}.map                          # atom-specific affinity map\n"

        gpf_content += """elecmap r.e.map                      # electrostatic potential map
dsolvmap r.d.map              # desolvation potential map
dielectric -0.1465                   # <0, AD4 distance-dep.diel;>0, constant
"""

        with open(os.path.join(sub_path, "r.gpf"), 'w') as gpf:
            gpf.write(gpf_content)

        print(f"✅ r.gpf ساخته شد برای: {subfolder}")
