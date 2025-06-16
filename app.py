from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import tempfile
import os
import utils.preprocessing as utils
import utils.model as model
import pandas as pd


app = Flask(__name__)
app.secret_key = 'segmentasi'


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    print(request.files['file'])

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = utils.preprocessing(file)

        hasil_df, silhouette, dbi = model.model(df)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", dir=".", mode='w+b') as tmp:
            hasil_df.to_excel(tmp.name, index=False)
            tmp.flush()
        session['silhouette'] = silhouette
        session['dbi'] = dbi
        session['download_filename'] = tmp.name 
        return redirect(url_for('result'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/result')
def result():

    silhouette = session.get('silhouette')
    dbi = session.get('dbi')
    download_link = f"/download/{session.get('download_filename')}"
    df_final = pd.read_excel(session.get('download_filename'))

    df_before = pd.read_csv("df_before.csv")
    if 'Nama Pelanggan' in df_final.columns and 'Nama Pelanggan' in df_before.columns:
        df_final['Nama Pelanggan'] = df_before['Nama Pelanggan']

    if 'Kapasitas Seat' in df_final.columns and 'Kapasitas Seat' in df_before.columns:
        df_final['Kapasitas Seat'] = df_before['Kapasitas Seat']

    cutpoints = df_final[['Length','Recency','Frequency','Monetary']].quantile([0.33, 0.66])
    df_final['L_Score'] = df_final['Length'].apply(
    lambda x: 'Low'    if x <= cutpoints.loc[0.33, 'Length']
              else ('Medium' if x <= cutpoints.loc[0.66, 'Length'] else 'High')
    )
    df_final['R_Score'] = df_final['Recency'].apply(
        lambda x: 'High'    if x <= cutpoints.loc[0.33, 'Recency']
                else ('Medium' if x <= cutpoints.loc[0.66, 'Recency'] else 'Low')
    )
    df_final['F_Score'] = df_final['Frequency'].apply(
        lambda x: 'Low'    if x <= cutpoints.loc[0.33, 'Frequency']
                else ('Medium' if x <= cutpoints.loc[0.66, 'Frequency'] else 'High')
    )
    df_final['M_Score'] = df_final['Monetary'].apply(
        lambda x: 'Low'    if x <= cutpoints.loc[0.33, 'Monetary']
                else ('Medium' if x <= cutpoints.loc[0.66, 'Monetary'] else 'High')
    )

    # gabungkan nilai LRFM per pelanggan
    df_final['LRFM_combo'] = (
        df_final['L_Score'] + ' Length, ' +
        df_final['R_Score'] + ' Recency, ' +
        df_final['F_Score'] + ' Frequency, ' +
        df_final['M_Score'] + ' Monetary'
    )


    # Mapping dictionary
    lrfm_to_cluster = {
        (('Medium'),('Medium','Low'), ('Low', 'Medium','High'), 'High'): 'High Value Loyal Customers',
        ('High', ('Medium','High'),('Medium','High'), 'High'): 'Platinum Customers',
        ('High', ('Low', 'Medium'), 'High', ('Low', 'Medium')): 'High Frequency Buying Customers',
        (('Medium', 'Low'), 'High', 'High', 'High'): 'Potential Loyal Customers',
        ('High', 'High', 'High', 'Low'): 'Potential Consumption Customers',
        ('High', 'High', ('Medium', 'Low'), ('Medium', 'Low','High')): 'Potential High Frequency Customers',
        ('Low', 'High', 'Low', 'Medium'): 'High Value New Customers',
        ('High', 'Low', 'Low', 'High'): 'Spender Promotion Customers',
        ('Low', ('Low', 'Medium','High'),('Low', 'Medium','High'), 'Low'): 'Uncertain New Customers',
        ('High', 'Low', 'Medium', 'High'): 'High Value Lost Customers',
        (('Low', 'Medium','High'), 'Low', 'Low', 'Low'): 'Consumption Lost Customers',
        ('Low', 'Low', 'Low', 'Low'): 'Low Consumption Cost Customers',
        ('High', 'High', 'Low', 'Low'): 'High Consumption Cost Customers',
        ('Medium', 'Medium', 'Medium', 'Medium'): 'Average Value Customers',
        ('Medium', 'High', 'Medium', 'Medium'): 'Moderate Potential Customers',
        ('Medium', 'Low', 'Medium', 'Medium'): 'At Risk Customers',
        ('Low', 'Medium', 'Medium', 'Medium'): 'New Emerging Customers',
        ('High', 'Medium', 'Medium', 'Medium'): 'Strong Potential Customers',
        ('Low', ('Low', 'Medium','High'), ('Low', 'Medium','High'), ('Medium','High')): 'New High Spender',
        'Other': 'Uncertain Customers',
    }

    # Lihat modus per cluster
    cluster_avgs = df_final.groupby('Cluster')[['L_Score', 'R_Score', 'F_Score', 'M_Score']].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'Uncertain'
    ).reset_index()

    # Concat nilai LRFM untuk cluster
    cluster_avgs['LRFM_combo'] = cluster_avgs['L_Score'].astype(str) + ' Length, ' + \
        cluster_avgs['R_Score'].astype(str) + ' Recency, ' + \
        cluster_avgs['F_Score'].astype(str) + ' Frequency, ' + \
        cluster_avgs['M_Score'].astype(str) + ' Monetary'

    # Map nama cluster
    def map_cluster_name(row):
        combo = (
            row['L_Score'],
            row['R_Score'],
            row['F_Score'],
            row['M_Score']
        )
        # Mencocokkan kombinasi skor LRFM dengan dictionary
        for key in lrfm_to_cluster.keys():
            if isinstance(key, tuple) and len(key) == 4:
                match = True
                for i, val in enumerate(key):
                    if isinstance(val, tuple):
                        if row[['L_Score','R_Score','F_Score','M_Score']].iloc[i] not in val:
                            match = False
                            break
                    else:
                        if row[['L_Score','R_Score','F_Score','M_Score']].iloc[i] != val:
                            match = False
                            break
                if match:
                    return lrfm_to_cluster[key]
        return lrfm_to_cluster.get('Other', 'Uncertain Customers')

    cluster_avgs['Cluster Name'] = cluster_avgs.apply(map_cluster_name, axis=1)

    cluster_name_map = dict(
        zip(cluster_avgs['Cluster'], cluster_avgs['Cluster Name'])
    )

    # Supaya tidak terjadi overwriting
    cluster_name_map = {k: f"{v} (Cluster {k})" for k, v in cluster_name_map.items()}

    # Siapkan data untuk HTML render
    selected_columns = ['Nama Pelanggan','L_Score', 'R_Score', 'F_Score', 'M_Score']
    cluster_tables = {}
    columns_to_display = [col for col in selected_columns if col != 'Cluster']

    for cluster_label in sorted(df_final['Cluster'].unique()):
        cluster_df = df_final[df_final['Cluster'] == cluster_label][columns_to_display].drop_duplicates(subset=['Nama Pelanggan'])
        base_name = cluster_name_map.get(cluster_label, f"Cluster {cluster_label}")
        try:
            cluster_label_plus_one = int(cluster_label) + 1
            cluster_name = base_name.replace(str(cluster_label), str(cluster_label_plus_one))
        except Exception:
            cluster_name = base_name

        cluster_tables[cluster_name] = cluster_df.to_html(
            classes="table table-striped table-bordered", index=False
        )
   
    df_final = df_final.drop(columns=[
        'Tipe Perjalanan', 'Length Trip', 'Total Spending',
        'Length', 'Recency', 'Frequency', 'Monetary'
    ], errors='ignore')  
    df_final = df_final.drop_duplicates(subset=['Nama Pelanggan']);
    # Simpan hasil segmentasi 
    tmp_filename = "processed_tmp.xlsx"
    df_final.to_excel(tmp_filename, index=False)


    return render_template(
        "result.html",
        silhouette_score=silhouette,
        dbi_score=dbi,
        # download_link=download_link,
        cluster_tables=cluster_tables
    )