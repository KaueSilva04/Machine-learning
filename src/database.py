import sqlite3
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'qrcode_data.db')

def conectar():
    """Retorna uma conexão aberta com o SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def inicializar_db():
    """Garante que todas as tabelas necessárias existam no banco de dados."""
    conn = conectar()
    c = conn.cursor()
    
    # Tabela 1: qr_extractions (Extrações individuais do Algoritmo Genético)
    c.execute('''CREATE TABLE IF NOT EXISTS qr_extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        filtros TEXT,
        score REAL,
        decoded_text TEXT,
        bbox TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Tabela 2: image_features (Características físicas das imagens do dataset)
    c.execute('''CREATE TABLE IF NOT EXISTS image_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT UNIQUE,
        brightness REAL,
        contrast REAL,
        saturation REAL,
        laplacian_variance REAL,
        edge_density REAL,
        qr_raw_detected INTEGER,
        qr_raw_text TEXT,
        width INTEGER,
        height INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Tabela 3: ga_experiments (Histórico das rodadas do Algoritmo Genético)
    c.execute('''CREATE TABLE IF NOT EXISTS ga_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        usar_rede_neural BOOLEAN,
        geracoes INTEGER,
        populacao INTEGER,
        tempo_total_segundos REAL,
        melhor_score_final REAL,
        media_score_final REAL,
        historico_geracoes TEXT -- Lista de dicionários em formato JSON
    )''')
    
    conn.commit()
    conn.close()

def salvar_experimento(usar_rede_neural, geracoes, populacao, tempo_total, melhor_score, media_score, historico_geracoes):
    """Insere o resumo de uma rodada de Algoritmo Genético para análise posterior."""
    conn = conectar()
    c = conn.cursor()
    c.execute('''
        INSERT INTO ga_experiments 
        (usar_rede_neural, geracoes, populacao, tempo_total_segundos, melhor_score_final, media_score_final, historico_geracoes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        1 if usar_rede_neural else 0,
        geracoes,
        populacao,
        tempo_total,
        melhor_score,
        media_score,
        json.dumps(historico_geracoes)
    ))
    conn.commit()
    conn.close()

def obter_comparativo_experimentos():
    """
    Retorna estatísticas comparativas entre as rodadas com e sem rede neural.
    Calcula a velocidade de convergência e ganhos médios.
    """
    conn = conectar()
    c = conn.cursor()
    
    # Obter médias agregadas para ambos os modos
    c.execute('''
        SELECT 
            usar_rede_neural, 
            COUNT(*) as total_rodadas,
            AVG(tempo_total_segundos) as tempo_medio,
            AVG(melhor_score_final) as melhor_medio,
            AVG(media_score_final) as media_geral
        FROM ga_experiments
        GROUP BY usar_rede_neural
    ''')
    resumos = c.fetchall()
    
    # OTIMIZAÇÃO: Obter o histórico de gerações do último experimento COM rede neural
    c.execute('''
        SELECT historico_geracoes 
        FROM ga_experiments 
        WHERE usar_rede_neural = 1 
        ORDER BY id DESC LIMIT 1
    ''')
    row_com = c.fetchone()
    
    # OTIMIZAÇÃO: Obter o histórico de gerações do último experimento SEM rede neural
    c.execute('''
        SELECT historico_geracoes 
        FROM ga_experiments 
        WHERE usar_rede_neural = 0 
        ORDER BY id DESC LIMIT 1
    ''')
    row_sem = c.fetchone()
    
    conn.close()
    
    stats = {
        'com_nn': {'total': 0, 'tempo_medio': 0.0, 'melhor_medio': 0.0, 'media_geral': 0.0, 'historico': []},
        'sem_nn': {'total': 0, 'tempo_medio': 0.0, 'melhor_medio': 0.0, 'media_geral': 0.0, 'historico': []},
        'ganho_melhor_score': 0.0,
        'ganho_tempo_porcento': 0.0
    }
    
    for row in resumos:
        usar_nn = bool(row[0])
        chave = 'com_nn' if usar_nn else 'sem_nn'
        stats[chave] = {
            'total': row[1],
            'tempo_medio': row[2] if row[2] else 0.0,
            'melhor_medio': row[3] if row[3] else 0.0,
            'media_geral': row[4] if row[4] else 0.0,
            'historico': []
        }
        
    if row_com and row_com[0]:
        try:
            stats['com_nn']['historico'] = json.loads(row_com[0])
        except Exception:
            stats['com_nn']['historico'] = []
            
    if row_sem and row_sem[0]:
        try:
            stats['sem_nn']['historico'] = json.loads(row_sem[0])
        except Exception:
            stats['sem_nn']['historico'] = []
            
    com = stats['com_nn']
    sem = stats['sem_nn']
    
    if com['total'] > 0 and sem['total'] > 0:
        stats['ganho_melhor_score'] = com['melhor_medio'] - sem['melhor_medio']
        if sem['tempo_medio'] > 0:
            # Ganho de velocidade (tempo economizado em porcentagem)
            stats['ganho_tempo_porcento'] = ((sem['tempo_medio'] - com['tempo_medio']) / sem['tempo_medio']) * 100.0
            
    return stats

def listar_todos(limite=50):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT id, image_name, score, decoded_text, timestamp FROM qr_extractions ORDER BY id DESC LIMIT ?', (limite,))
    rows = c.fetchall()
    conn.close()
    return rows

def buscar_por_imagem(nome):
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT id, image_name, filtros, score, decoded_text, bbox, timestamp FROM qr_extractions WHERE image_name LIKE ? ORDER BY id DESC', (f'%{nome}%',))
    rows = c.fetchall()
    conn.close()
    return rows

def excluir_id(uid):
    conn = conectar()
    c = conn.cursor()
    c.execute('DELETE FROM qr_extractions WHERE id = ?', (uid,))
    conn.commit()
    afetados = c.rowcount
    conn.close()
    return afetados

def compactar_db():
    conn = conectar()
    conn.execute('VACUUM')
    conn.close()

def obter_estatisticas_gerais():
    """Retorna um resumo estatístico em formato dict para a API Web."""
    conn = conectar()
    c = conn.cursor()
    
    # Total de extrações
    c.execute('SELECT COUNT(*) FROM qr_extractions')
    total_ext = c.fetchone()[0]
    
    # Extrações com sucesso total (score = 1.0)
    c.execute('SELECT COUNT(*) FROM qr_extractions WHERE score = 1.0')
    sucesso_total = c.fetchone()[0]
    
    # Extrações com sucesso parcial (score = 0.5)
    c.execute('SELECT COUNT(*) FROM qr_extractions WHERE score = 0.5')
    sucesso_parcial = c.fetchone()[0]
    
    # Total de imagens físicas catalogadas
    c.execute('SELECT COUNT(*) FROM image_features')
    total_imgs = c.fetchone()[0]
    
    # Imagens que decodificam de forma pura (sem filtros)
    c.execute('SELECT COUNT(*) FROM image_features WHERE qr_raw_detected = 1')
    imgs_pura_ok = c.fetchone()[0]
    
    # Total de experimentos realizados
    c.execute('SELECT COUNT(*) FROM ga_experiments')
    total_exps = c.fetchone()[0]
    
    conn.close()
    
    taxa_sucesso_geral = (sucesso_total / total_ext * 100.0) if total_ext > 0 else 0.0
    
    return {
        'total_extractions': total_ext,
        'sucesso_total': sucesso_total,
        'sucesso_parcial': sucesso_parcial,
        'total_imagens': total_imgs,
        'imagens_puras_decodificadas': imgs_pura_ok,
        'total_experimentos': total_exps,
        'taxa_sucesso_geral': round(taxa_sucesso_geral, 2)
    }

def exportar_csv(arquivo):
    import csv
    conn = conectar()
    c = conn.cursor()
    c.execute('SELECT * FROM qr_extractions ORDER BY id')
    rows = c.fetchall()
    colunas = [d[0] for d in c.description]
    
    with open(arquivo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(colunas)
        writer.writerows(rows)
    conn.close()
