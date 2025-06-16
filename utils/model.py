from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from bayes_opt import BayesianOptimization

from minisom import MiniSom


def model(df):
    best_params =    {'learning_rate': 0.23097867011994386, 'sigma':0.9753571532049581, 'x_dim': 1, 'y_dim': 3}
    X = df.drop(columns=['Nama Pelanggan', 'Tanggal',"Status","Status Pembayaran"])
    X = X.values

    # def som_objective(learning_rate, sigma, x_dim, y_dim):
    #     x_dim, y_dim = int(x_dim), int(y_dim)

    #     som = MiniSom(x_dim, y_dim, input_len=X.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=1)
    #     som.random_weights_init(X)
    #     som.train_random(X, num_iteration=20000)

    #     labels = [som.winner(x) for x in X]
    #     labels = [l[0] * x_dim + l[1] for l in labels]
    #     num_clusters = len(set(labels))

    #     if num_clusters > 10:
    #         return -1

    #     return silhouette_score(X, labels)

    # param_bounds = {
    #     'learning_rate': (0.01, 0.6),
    #     'sigma': (0.5, 1.0),
    #     'x_dim': (1, 2),
    #     'y_dim': (2, 5)
    # }

    # optimizer = BayesianOptimization(
    #     f=som_objective,
    #     pbounds=param_bounds,
    #     random_state=42,
    #     verbose=2
    # )
    # optimizer.maximize(init_points=20, n_iter=30)
    # best_params = optimizer.max['params']
    # best_params['x_dim'] = int(best_params['x_dim'])
    # best_params['y_dim'] = int(best_params['y_dim'])
    
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
