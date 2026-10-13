import os

# مسیر فولدر مادر
root_folder = r"C:\Users\lenovo\Desktop\New folder"

# اسم فایلی که می‌خوای پاک شه
target_filename = "r.gpf"  # <-- اسم فایل موردنظر

def delete_target_files(root_path, filename):
    deleted_files = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        if filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"✅ Deleted: {file_path}")
                deleted_files += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
    print(f"\n🔍 Finished. Total files deleted: {deleted_files}")

# اجرای تابع
delete_target_files(root_folder, target_filename)
