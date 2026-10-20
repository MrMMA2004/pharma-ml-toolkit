import pandas as pd

# فایل‌ها
df_pic50 = pd.read_csv('Morgan_RDKIT_pIC50_5HT2A.csv')
df_pec50 = pd.read_csv('Morgan_RDKIT_pEC50_5HT2A.csv')

# ستون‌هایی که باید اول باشن
first_cols = ['SMILES', 'State', 'pIC50/pEC50']

# ----------------------
# مرتب‌سازی df_pic50
df_pic50_first = df_pic50[first_cols]
df_pic50_rest = df_pic50.drop(columns=first_cols)
df_pic50_rest = df_pic50_rest.reindex(sorted(df_pic50_rest.columns), axis=1)
df_pic50 = pd.concat([df_pic50_first, df_pic50_rest], axis=1)

# ----------------------
# مرتب‌سازی df_pec50
df_pec50_first = df_pec50[first_cols]
df_pec50_rest = df_pec50.drop(columns=first_cols)
df_pec50_rest = df_pec50_rest.reindex(sorted(df_pec50_rest.columns), axis=1)
df_pec50 = pd.concat([df_pec50_first, df_pec50_rest], axis=1)

# ----------------------
# ادغام نهایی با حفظ همه ستون‌ها
df_merged = pd.concat([df_pic50, df_pec50], axis=0, ignore_index=True, sort=False)

# ذخیره فایل
df_merged.to_csv('Total_5HT2A.csv', index=False)
