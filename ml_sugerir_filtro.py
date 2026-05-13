import sqlite3
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DB_PATH = 'qrcode_data.db'
MODEL_PATH = 'filtro_predictor.pkl'

FILTER_KEYS = ['kernel_size', 'contrast', 'bright', 'sat', 'sharp', 'clahe', 'thresh_block', 'thresh_c', 'gamma', 'denoise', 'expo']
IMAGE_KEYS = ['brightness', 'contrast', 'saturation', 'laplacian_variance', 'edge_density', 'qr_raw_detected', 'width', 'height']


def carregar_base():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT e.image_name, e.filtros, e.score,
               f.brightness, f.contrast, f.saturation, f.laplacian_variance, f.edge_density, f.qr_raw_detected, f.width, f.height
        FROM qr_extractions e
        JOIN image_features f ON e.image_name = f.image_name
        WHERE e.filtros IS NOT NULL
    ''')
    linhas = c.fetchall()
    conn.close()

    X = []
    y = []
    for row in linhas:
        filtros_json = row[1]
        score = row[2]
        img_vals = list(row[3:])
        try:
            filtros = json.loads(filtros_json)
        except Exception:
            continue

        if not all(k in filtros for k in FILTER_KEYS):
            continue

        filter_vals = [filtros[k] for k in FILTER_KEYS]
        X.append(img_vals + filter_vals)
        y.append(score)

    return np.array(X, dtype=float), np.array(y, dtype=float)


def treinar_modelo(test_size=0.2, random_state=42):
    X, y = carregar_base()
    if len(X) < 20:
        print('Dados insuficientes para treinar (', len(X), 'amostras).')
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=random_state))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse:.6f} | R2: {r2:.6f} | samples: {len(X)}')

    joblib.dump(model, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    return model


def carregar_modelo():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return treinar_modelo()


def sugerir_filtros(caract, candidatos, top_n=5):
    # caract: dict com IMAGE_KEYS
    # candidatos: lista de dicts com FILTER_KEYS
    model = carregar_modelo()
    if model is None:
        raise RuntimeError('Modelo não disponível')

    feature_img = [caract.get(k, 0) for k in IMAGE_KEYS]
    X_test = []
    candidate_list = []

    for c in candidatos:
        base = feature_img[:]
        fvals = [c.get(k, 0) for k in FILTER_KEYS]
        X_test.append(base + fvals)
        candidate_list.append(c)

    X_test = np.array(X_test, dtype=float)
    pred = model.predict(X_test)

    ranked = sorted(zip(candidate_list, pred), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def main():
    print('Treinando modelo de filtro com dados existentes...')
    treinar_modelo()

    # Exemplo de uso com 2 filtros de teste
    sample_caract = {
        'brightness': 120, 'contrast': 40, 'saturation': 100, 'laplacian_variance': 120, 'edge_density': 0.01,
        'qr_raw_detected': 0, 'width': 1280, 'height': 720
    }
    candidates = [
        {'kernel_size': 3, 'contrast': 120, 'bright': 120, 'sat': 110, 'sharp': 30, 'clahe': 5, 'thresh_block': 15, 'thresh_c': 2, 'gamma': 100, 'denoise': 0, 'expo': 100},
        {'kernel_size': 5, 'contrast': 90, 'bright': 140, 'sat': 90, 'sharp': 50, 'clahe': 10, 'thresh_block': 25, 'thresh_c': 3, 'gamma': 90, 'denoise': 5, 'expo': 110},
    ]

    recomendacoes = sugerir_filtros(sample_caract, candidates, top_n=2)
    print('\nTop filtros sugeridos:')
    for f, score in recomendacoes:
        print(f'  score previsto {score:.4f} ->', f)


if __name__ == '__main__':
    main()
