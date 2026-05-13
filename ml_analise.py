import sqlite3
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def carregar_dados():
    conn = sqlite3.connect('qrcode_data.db')
    c = conn.cursor()
    c.execute("SELECT filtros, score FROM qr_extractions WHERE score IS NOT NULL")
    dados = c.fetchall()
    conn.close()
    
    X = []
    y = []
    for filtros_json, score in dados:
        filtros = json.loads(filtros_json)
        X.append([filtros.get(k, 0) for k in ['kernel_size', 'contrast', 'bright', 'sat', 'sharp', 'clahe', 'thresh_block', 'thresh_c', 'gamma', 'denoise', 'expo']])
        y.append(score)
    
    return X, y

def treinar_modelo(X, y):
    if len(X) < 10:
        print("Poucos dados para treinar modelo.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    
    # Importância das features
    features = ['kernel_size', 'contrast', 'bright', 'sat', 'sharp', 'clahe', 'thresh_block', 'thresh_c', 'gamma', 'denoise', 'expo']
    importancias = model.feature_importances_
    for feat, imp in zip(features, importancias):
        print(f"{feat}: {imp:.4f}")
    
    return model

if __name__ == "__main__":
    X, y = carregar_dados()
    print(f"Dados carregados: {len(X)} amostras")
    
    if len(X) > 0:
        model = treinar_modelo(X, y)
    else:
        print("Nenhum dado encontrado. Execute processamento.py primeiro.")