from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from minisom import MiniSom


def model(df):
    # best_params =   {'learning_rate': 0.13093638136374644, 'sigma': 0.6720324493442158, 'x_dim': 1, 'y_dim': 3}
    best_params =    {'learning_rate': 0.23097867011994386, 'sigma':0.9753571532049581, 'x_dim': 1, 'y_dim': 3}
    # best_params =  {'learning_rate': 0.18743733946949004, 'sigma': 0.6139493860652041, 'x_dim': 1, 'y_dim': 3}
    X = df.drop(columns=['Nama Pelanggan', 'Tanggal',"Status","Status Pembayaran"])
    X = X.values
    
    seed = 1

    som = MiniSom(
        x=int(best_params['x_dim']),
        y=int(best_params['y_dim']),
        input_len=X.shape[1],
        sigma=best_params['sigma'],
        learning_rate=best_params['learning_rate'],
        random_seed=seed
    )

    som.random_weights_init(X)
    som.train_random(X, num_iteration=20000)

    labels = [som.winner(x) for x in X]
    labels = [l[0] * int(best_params['x_dim']) + l[1] for l in labels]

    score = round(silhouette_score(X, labels), 2)
    dbi = round(davies_bouldin_score(X, labels), 2)

    hasil_df = df.copy()
    hasil_df['Cluster'] = labels

    return hasil_df, score, dbi
