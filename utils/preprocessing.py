import tempfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from flask import session
# import json



def preprocessing(file):
    with tempfile.NamedTemporaryFile(delete=False, dir="dataset", suffix=".xls") as tmp:
        tmp_path = tmp.name

    file.save(tmp_path)
    df = pd.read_excel(tmp_path)
    df.drop_duplicates(inplace=True)
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
    df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

    df['Asal'] = df['Asal'].replace({
        "Trawas,UTC": "UTC", "Pacet (GKI)": "GKI Pacet",
        "MALANG (BANDULAN PAS AS": "Malang", "Malang Kota": "Malang",
        "Batu (Syalom)": "Batu", "Sby": "Surabaya"
    })

    if 'Harga' in df.columns:
        df['Harga'] = df['Harga'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')

    if 'Tujuan' in df.columns:
        split_values = df['Tujuan'].str.split()
        df['Tujuan'] = split_values.str[0]
        df['Tipe Perjalanan'] = split_values.str[-1]

    # # Convert 'Berangkat' dan 'Tiba' ke datetime
    df['Berangkat'] = pd.to_datetime(df['Berangkat'], errors='coerce')
    df['Tiba'] = pd.to_datetime(df['Tiba'], errors='coerce')

    # # Hitung lama trip, satuannya jam
    # df['Length Trip'] = (df['Tiba'] - df['Berangkat']).dt.total_seconds() / 3600
    # df['Length Trip'] = df['Length Trip'].clip(lower=0)
    # df['Avg Length Trip'] = df.groupby('Nama Pelanggan')['Length Trip'].transform('mean')

    # # # Hitung avg pengeluaran per pelanggan
    # df['Avg Spending'] = df.groupby('Nama Pelanggan')['Harga'].transform('mean')

    # # # # Hitung jumlah perjalanan per pelanggan
    # df['Trip Count'] = df.groupby('Nama Pelanggan')['Harga'].transform('count')

    # # # # Hitung rata-rata harga tiket
    # df['Total Spending'] = df.groupby('Nama Pelanggan')['Harga'].transform('sum')
    # df['Avg Ticket Price'] = df['Total Spending'] / df['Trip Count']

    df = df[(df['Status'] == 'DEAL') & (df['Status Pembayaran'] == 'LUNAS')]

    df.drop(columns=['Kapasitas Seat',"Kodenota", "NoPlat", "Berangkat", "Tiba", "PPN", "Asal","Tujuan"], inplace=True)

    # supaya index ga berantakan harus di reset
    df.sort_values(by='Tanggal', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Hitung Recency
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', errors='coerce')
    df_latest = df.groupby('Nama Pelanggan')['Tanggal'].max()
    max_date = df['Tanggal'].max()
    df_recency = (max_date - df_latest).dt.days.to_frame(name='Recency')

    # Hitung Length
    df_earliest = df.groupby('Nama Pelanggan')['Tanggal'].min()
    df_length = (max_date - df_earliest).dt.days.to_frame(name='Length')

    # Hitung Frequency
    df_frequency = df.groupby('Nama Pelanggan')['Nama Pelanggan'].count().to_frame(name='Frequency')

    # Hitung Monetary
    df_monetary = df.groupby('Nama Pelanggan')[
        'Harga'].sum().to_frame(name='Monetary')

    # Penggabungan data lrfm
    df_lrfm = pd.concat([df_length, df_recency, df_frequency, df_monetary], axis=1)
    df.drop_duplicates(subset=['Nama Pelanggan'], inplace=True)

    # Penggabungan dengan df awal
    df_final = df.merge(df_lrfm, on='Nama Pelanggan', how='left').copy()
    print(df_final['Frequency'].value_counts().sort_index())


    # Min-Max Normalization
    num_columns = df_final.select_dtypes(include=['number']).columns
    if not num_columns.empty:
        scaler = MinMaxScaler()
        df_final[num_columns] = scaler.fit_transform(df_final[num_columns])

    # Lakukan encoding
    categorical_columns = df_final.select_dtypes(include=['object']).columns
    label_encoder_classes = {}
    label_encoder = LabelEncoder()
    df_before = df_final.copy()

    for col in categorical_columns:
        df_final[col] = label_encoder.fit_transform(df_final[col].astype(str))
        label_encoder_classes[col] = label_encoder.classes_.tolist()
    
    df_before.to_csv("df_before.csv", index=False)

    return df_final


def processss(file):
    preprocessed_df = preprocessing(file)
    return preprocessed_df
