import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DB_PATH = 'qrcode_data.db'


def carregar_dados():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT f.brightness, f.contrast, f.saturation, f.laplacian_variance, f.edge_density,
               f.qr_raw_detected, f.width, f.height, e.score
        FROM image_features f
        JOIN qr_extractions e ON f.image_name = e.image_name
    ''')
    linhas = c.fetchall()
    conn.close()

    if not linhas:
        print('Nenhuma entrada encontrada para treinar ML.')
        return None, None

    X = [list(l[:8]) for l in linhas]
    y = [l[8] for l in linhas]
    return X, y


def treinar_modelo():
    X, y = carregar_dados()
    if X is None or len(X) < 10:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.6f}, R2: {r2:.6f}')

    importance = model.feature_importances_
    features = ['brightness', 'contrast', 'saturation', 'laplacian_variance', 'edge_density', 'qr_raw_detected', 'width', 'height']
    print('Feature importances:')
    for f, imp in zip(features, importance):
        print(f'  {f}: {imp:.4f}')


if __name__ == '__main__':
    treinar_modelo()
