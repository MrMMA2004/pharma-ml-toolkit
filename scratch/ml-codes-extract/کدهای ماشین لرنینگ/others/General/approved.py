import pandas as pd
from docx import Document

# تنظیمات
drug_column = "drug"  # ستون نام دارو
approval_column = "regulatory approval"  # ستون وضعیت تأیید
output_file = "approved_drugs.docx"

# ایجاد یک سند ورد جدید
doc = Document()

# پردازش 17 فایل
for i in range(1, 18):
    file_name = f"{i}.tsv"
    try:
        df = pd.read_csv(file_name, sep="\t")  # خواندن فایل TSV

        # اصلاح نام ستون‌ها (حذف فاصله‌ها و کوچک‌سازی)
        df.columns = df.columns.str.strip().str.lower()

        # بررسی وجود ستون‌ها
        if drug_column in df.columns and approval_column in df.columns:
            # فیلتر کردن فقط داروهایی که 'Approved' هستند
            approved_drugs = df[df[approval_column].str.strip().str.lower() == "approved"]
            for _, row in approved_drugs.iterrows():
                drug_name = str(row[drug_column]).strip()
                doc.add_paragraph(drug_name)  # فقط نام داروها را ذخیره می‌کند
        else:
            print(f"⚠️ هشدار: ستون‌های '{drug_column}' یا '{approval_column}' در '{file_name}' یافت نشد.")

    except Exception as e:
        print(f"❌ خطا در پردازش {file_name}: {e}")

# بررسی اینکه آیا داده‌ای در فایل ورد نوشته شده یا نه
if len(doc.paragraphs) > 0:
    doc.save(output_file)
    print(f"✅ فایل ورد '{output_file}' با موفقیت ایجاد شد! (فقط داروهای Approved)")
else:
    print("⚠️ هیچ داروی Approved یافت نشد!")
