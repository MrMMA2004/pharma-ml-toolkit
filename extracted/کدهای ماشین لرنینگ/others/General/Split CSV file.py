import pandas as pd

def split_csv(input_file, output_prefix, chunk_size=1_500_000):
    # خواندن فایل به صورت chunk شده
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iter):
        output_file = f"{output_prefix}_part{i + 1}.csv"
        chunk.to_csv(output_file, index=False, encoding='utf-8')  # Added encoding
        print(f"Saved {output_file} with {len(chunk)} rows")

# مثال استفاده:
split_csv("6_p2.csv", r"C:\Users\ASMAN RAYANEH\Desktop\abdoos")
