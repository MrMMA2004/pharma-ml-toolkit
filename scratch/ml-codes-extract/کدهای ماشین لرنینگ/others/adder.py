import os
import shutil

# مسیر فایل‌های اصلی autodock4 و autogrid4
source_folder = r"C:\Users\lenovo\Desktop\CTNAP2"
autodock_file = os.path.join(source_folder, "autodock4.exe")
autogrid_file = os.path.join(source_folder, "autogrid4.exe")
receptor_file = os.path.join(source_folder, "r.pdbqt")
# پوشه‌ای که می‌خوای این دوتا فایل برن تو همه‌ی زیرفولدرهاش
main_folder = r"C:\Users\lenovo\Desktop\CTNAP2"

# رفتن به همه‌ی زیرپوشه‌ها و کپی فایل‌ها
for root, dirs, files in os.walk(main_folder):
    for d in dirs:
        target_path = os.path.join(root, d)
        print(f"📂 کپی به: {target_path}")
        try:
            shutil.copy(autodock_file, target_path)
            shutil.copy(autogrid_file, target_path)
            shutil.copy(receptor_file, target_path)
        except Exception as e:
            print(f"❌ خطا در کپی به {target_path}: {e}")
