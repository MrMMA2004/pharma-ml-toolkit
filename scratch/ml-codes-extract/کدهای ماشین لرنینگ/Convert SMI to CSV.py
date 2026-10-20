import csv
from tqdm import tqdm

def smi_to_csv_stream(smi_path, output_csv):
    """
    خواندن خط‌به‌خط یک فایل .smi (هر خط: <SMILES> <ID>) و
    نوشتن مستقیم آن در یک فایل CSV با نمایش نوار پیشرفت.
    به این صورت که هیچ لیستی از پیش در حافظه ذخیره نمی‌شود.
    """
    # ۱. باز کردن فایل CSV جهت نوشتن (با BOM برای سازگاری بهتر با Excel)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as out_f:
        writer = csv.writer(out_f)
        # نوشتن هدر (ستون‌ها)
        writer.writerow(["SMILES", "ID"])

        # ۲. خواندن خط‌به‌خط فایل .smi و نوشتن فوری در CSV
        with open(smi_path, "r", encoding="utf-8", errors="ignore") as in_f:
            for line in tqdm(in_f, desc="در حال خواندن و نوشتن", unit="line"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                smi = parts[0]
                ident = " ".join(parts[1:]) if len(parts) > 1 else ""
                writer.writerow([smi, ident])

    print(f"\n✅ تمام داده‌ها در فایل CSV ذخیره شد:\n  {output_csv}")


if __name__ == "__main__":
    input_smi = '6_p1.smi'
    output_csv = "6_p1.csv"

    print("شروع تبدیل فایل .smi به CSV (خط‌به‌خط) با نمایش نوار پیشرفت...\n")
    smi_to_csv_stream(input_smi, output_csv)