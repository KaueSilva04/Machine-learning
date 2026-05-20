import sqlite3
import json
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.database import conectar
from src.filters import FILTER_KEYS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'filtro_predictor.pkl')

IMAGE_KEYS = ['brightness', 'contrast', 'saturation', 'laplacian_variance', 'edge_density', 'qr_raw_detected', 'width', 'height']

def carregar_base():
    """Carrega dados históricos de extrações e características físicas para treino da MLP."""
    conn = conectar()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT e.image_name, e.filtros, e.score,
                   f.brightness, f.contrast, f.saturation, f.laplacian_variance, f.edge_density, f.qr_raw_detected, f.width, f.height
            FROM qr_extractions e
            JOIN image_features f ON e.image_name = f.image_name
            WHERE e.filtros IS NOT NULL AND e.score IS NOT NULL
        ''')
        linhas = c.fetchall()
    except sqlite3.OperationalError:
        linhas = []
    finally:
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

        # Garante retrocompatibilidade se as chaves limpas estiverem contidas nos filtros salvos
        if not all(k in filtros for k in FILTER_KEYS):
            continue

        filter_vals = [filtros[k] for k in FILTER_KEYS]
        
        # Concatena características físicas da imagem com as configurações de filtros
        X.append(img_vals + filter_vals)
        y.append(score)

    return np.array(X, dtype=float), np.array(y, dtype=float)

def treinar_modelo(test_size=0.2, random_state=42):
    """
    Treina a rede neural MLPRegressor com a base de dados atual,
    salvando os pesos otimizados em models/filtro_predictor.pkl.
    """
    X, y = carregar_base()
    if len(X) < 20:
        print(f"⚠️ Amostras insuficientes para treinar o modelo ({len(X)} encontradas, mínimo de 20).")
        return None

    # Pipeline que escala as entradas e aplica a MLPRegressor
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            max_iter=1000, 
            random_state=random_state
        ))
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ MLP Treinada com Sucesso | Amostras: {len(X)} | MSE: {mse:.6f} | R2: {r2:.6f}")

        # Salva o arquivo de pesos serializado
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model
    except Exception as e:
        print(f"❌ Erro ao treinar o modelo: {e}")
        return None

def carregar_modelo():
    """Tenta carregar o modelo treinado. Se não existir, dispara o treino automático."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
            
    print("⏳ Carregando ou treinando modelo preditivo de filtros pela primeira vez...")
    return treinar_modelo()

def sugerir_filtros(caract, candidatos, top_n=5):
    """
    Usa a IA treinada para pontuar um conjunto de candidatos a filtros de imagem,
    retornando os TOP N mais promissores baseado nas características físicas.
    """
    model = carregar_modelo()
    if model is None:
        # Se a IA não pôde ser carregada ou treinada, retorna os candidatos sem ordenação
        return [(c, 1.0) for c in candidatos[:top_n]]

    feature_img = [caract.get(k, 0) for k in IMAGE_KEYS]
    X_test = []
    candidate_list = []

    for c in candidatos:
        base = feature_img[:]
        fvals = [c.get(k, 0) for k in FILTER_KEYS]
        X_test.append(base + fvals)
        candidate_list.append(c)

    X_test = np.array(X_test, dtype=float)
    
    try:
        pred = model.predict(X_test)
        # Ordena do maior score previsto para o menor
        ranked = sorted(zip(candidate_list, pred), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    except Exception as e:
        print(f"⚠️ Erro ao realizar predição com a MLP: {e}")
        return [(c, 1.0) for c in candidatos[:top_n]]
