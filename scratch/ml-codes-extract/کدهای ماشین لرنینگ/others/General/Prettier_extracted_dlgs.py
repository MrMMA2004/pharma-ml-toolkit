import pandas as pd

# فایل اکسل را بخوان
df = pd.read_excel("cluster_summary.xlsx")

# استخراج عدد از ستون "DLG File"
df["DLG_Number"] = df["DLG File"].astype(str).str.extract(r"(\d+)")[0]

# فقط ردیف‌هایی که واقعاً عدد دارن
df = df[df["DLG_Number"].notna()]

# تبدیل به عدد صحیح
df["DLG_Number"] = df["DLG_Number"].astype(int)

# ست کردن DLG_Number به عنوان index
df.set_index("DLG_Number", inplace=True)

# ساخت یک DataFrame با اندیس‌های 0 تا 300
full_index = pd.DataFrame(index=range(0, 301))

# ادغام بر اساس ایندکس
merged = full_index.merge(df, left_index=True, right_index=True, how='left')

# ذخیره فایل نهایی
merged.to_excel("aligned_output.xlsx", index=True)
