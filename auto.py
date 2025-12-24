import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import sys

# Konfigurasi Nama File
INPUT_FILE = '2020-2025.csv'
OUTPUT_FILE = 'data_gdp_asean_clean.csv'

# Daftar Negara ASEAN (Sesuai dengan yang ada di Notebook)
# Catatan: "Philipines" ditulis sesuai data asli di notebook (typo dari Philippines)
ASEAN_COUNTRIES = [
    "Indonesia", "Malaysia", "Singapore", "Myanmar", "Brunei", 
    "Philipines", "Thailand", "Cambodia", "Laos", "Vietnam"
]

def load_dataset(filepath):
    """Memuat dataset dari CSV."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File {filepath} tidak ditemukan.")
        print("Pastikan file dataset '2020-2025.csv' sudah ada di folder yang sama.")
        sys.exit(1)
    
    print(f"[INFO] Memuat data dari {filepath}...")
    df = pd.read_csv(filepath)
    return df

def filter_asean(df):
    """Menyaring data khusus negara ASEAN."""
    print("[INFO] Memfilter negara ASEAN...")
    df_filtered = df[df['Country'].isin(ASEAN_COUNTRIES)].copy()
    print(f"       Negara ditemukan: {df_filtered['Country'].unique()}")
    return df_filtered

def transform_to_long_format(df):
    """Mengubah format wide (tahun di kolom) menjadi long."""
    print("[INFO] Mengubah format data (Melting)...")
    # Kolom tahun yang tersedia di dataset
    year_columns = ["2020", "2021", "2022", "2023", "2024", "2025"]
    
    df_long = df.melt(
        id_vars="Country",
        value_vars=year_columns,
        var_name="Year",
        value_name="GDP"
    )
    
    # Pastikan Year menjadi integer
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

def preprocess_features(df):
    """Melakukan encoding, scaling, dan lag features."""
    print("[INFO] Melakukan preprocessing fitur...")
    
    # 1. Label Encoding untuk Country
    le = LabelEncoder()
    df['Country_Code'] = le.fit_transform(df['Country'])
    
    # 2. MinMax Scaling untuk GDP
    scaler = MinMaxScaler()
    df['GDP_Scaled'] = scaler.fit_transform(df[['GDP']])
    
    # 3. Sorting data agar urut berdasarkan Negara dan Tahun
    df = df.sort_values(by=['Country', 'Year'])
    
    # 4. Membuat Feature Lag (GDP Tahun Sebelumnya)
    # Shift sebanyak 1 tahun ke bawah per grup negara
    df['GDP_Prev_Year'] = df.groupby('Country')['GDP_Scaled'].shift(1)
    
    return df

def clean_and_save(df, output_path):
    """Menghapus NaN dan menyimpan file."""
    print("[INFO] Membersihkan NaN values...")
    # Baris pertama setiap negara akan menjadi NaN karena shift(1), kita drop
    df_clean = df.dropna()
    
    print(f"[INFO] Menyimpan data bersih ke {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print("[SUCCESS] Proses selesai!")
    print(df_clean.head())

def main():
    # 1. Load
    df = load_dataset(INPUT_FILE)
    
    # 2. Filter
    df_asean = filter_asean(df)
    
    # 3. Transform
    df_long = transform_to_long_format(df_asean)
    
    # 4. Preprocess (Encode, Scale, Feature Engineering)
    df_processed = preprocess_features(df_long)
    
    # 5. Clean & Save
    clean_and_save(df_processed, OUTPUT_FILE)

if __name__ == "__main__":
    main()