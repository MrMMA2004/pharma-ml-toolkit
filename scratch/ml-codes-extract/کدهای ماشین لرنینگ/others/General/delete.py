import os
import re

main_folder = r"/home/your_username/Projects/myfolder"

# فایل‌های مجاز
allowed_files = {
    "r.pdbqt",
    "r.dpf",
    "r.gpf",
    "autogrid4.exe",
    "autodock4.exe"
}
# regex برای فایل‌های 12.pdbqt، 1.pdbqt و غیره
allowed_pattern = re.compile(r"^\d+\.pdbqt$")

for root, dirs, files in os.walk(main_folder, topdown=False):
    for file in files:
        file_path = os.path.join(root, file)
        # بررسی اینکه فایل مجاز هست یا نه
        if file in allowed_files or allowed_pattern.match(file):
            print(f"✅ نگه‌داشتن فایل: {file_path}")
            continue
        try:
            print(f"🗑 حذف فایل: {file_path}")
            os.remove(file_path)
        except Exception as e:
            print(f"❌ خطا در حذف فایل {file_path}: {e}")

    # حذف پوشه‌های خالی بعد از حذف فایل‌ها
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):
            try:
                print(f"🧹 حذف پوشه خالی: {dir_path}")
                os.rmdir(dir_path)
            except Exception as e:
                print(f"❌ خطا در حذف پوشه {dir_path}: {e}")
