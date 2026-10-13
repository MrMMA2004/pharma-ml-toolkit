import pandas as pd

# مسیر فایل‌ها
file1 = "file2.xlsx"
file2 = "file1.xlsx"

# نام ستون‌ها (می‌تونی بر اساس نام یا اندیس عددی مشخص کنی)
column_name = "نام"  # یا مثلا column_name = 0 اگر فایل‌ها بدون نام ستون هستند

# خواندن فایل‌ها
df1 = pd.read_excel(file1, engine='openpyxl')
df2 = pd.read_excel(file2, engine='openpyxl')

# استخراج مقادیر یکتا از ستون
set1 = set(df1[column_name].dropna().astype(str).str.strip())
set2 = set(df2[column_name].dropna().astype(str).str.strip())

# پیدا کردن اشتراک
common = set1 & set2

# تبدیل به DataFrame برای ذخیره یا نمایش
result_df = pd.DataFrame(list(common), columns=["کلمات مشترک"])

# نمایش یا ذخیره
print(result_df)

# ذخیره در فایل اکسل
result_df.to_excel("مشترک.xlsx", index=False)
