import pandas as pd
import numpy as np

# خواندن فایل اکسل
df = pd.read_excel('xxx.xlsx')

# بررسی وجود ستون
if 'IC50 (nM)' not in df.columns:
    raise ValueError("ستون 'IC50 (nM)' یافت نشد.")

# حذف ردیف‌های خالی
df = df[df['IC50 (nM)'].notnull()]

# پاکسازی و تبدیل
def clean_ic50(value):
    try:
        val_str = str(value).strip()
        if val_str.startswith(('<', '>')):
            val_str = val_str[1:].strip()
        return float(val_str)
    except:
        return np.nan

# اعمال پاکسازی
df['IC50_clean'] = df['IC50 (nM)'].apply(clean_ic50)

# حذف مقادیر غیر عددی یا منفی یا صفر
df = df[df['IC50_clean'] > 0]

# محاسبه pIC50
df['pIC50'] = 9 - np.log10(df['IC50_clean'])

# ذخیره نتیجه
df.to_excel('output_with_pIC50.xlsx', index=False)

print("✅ مقادیر علمی، علامت‌دار و pIC50 با موفقیت پردازش شدند.")
