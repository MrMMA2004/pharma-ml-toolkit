import pandas as pd
import numpy as np
from tqdm import tqdm

# مسیر فایل ورودی و خروجی
input_path = "C:/Users/NP/Desktop/PYTHON/MordredDescriptors_Batched.xlsx"
output_path = "C:/Users/NP/Desktop/PYTHON/numeric_cells_only.xlsx"

# خواندن فایل
df = pd.read_excel(input_path)

# نمایش پیشرفت هنگام پاک‌سازی سلول‌ها
tqdm.pandas(desc="Cleaning non-numeric values")
df_cleaned = df.progress_applymap(lambda x: pd.to_numeric(x, errors='coerce') if not pd.isnull(x) else x)

# حذف ستون‌هایی که تمام مقادیرشان NaN شده
df_cleaned = df_cleaned.dropna(axis=1, how='all')

# ذخیره فایل خروجی
df_cleaned.to_excel(output_path, index=False)
print(f"✅ Saved cleaned data with only numeric cells to: {output_path} — shape: {df_cleaned.shape}")
