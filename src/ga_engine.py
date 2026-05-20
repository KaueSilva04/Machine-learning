import os
import cv2
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import time
import sqlite3
import json

from src.database import conectar, salvar_experimento, inicializar_db
from src.filters import aplicar_filtros, FILTER_KEYS
from src.decoders import decode_qr_image
from src.ml_model import carregar_modelo, sugerir_filtros, IMAGE_KEYS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "QRCode_diaADia")
TAXA_MUT = 0.7 
ELITE = 5
USAR_CUDA = False
SEED = 42

GLOBAL_DATASET = []

def calcular_caracteristicas_imagem(img):
    """Extrai características físicas de uma imagem."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:,:,1]))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / (w * h))

    decoded_text, _ = decode_qr_image(img)
    qr_detected = 1 if decoded_text else 0

    return {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'laplacian_variance': laplacian_variance,
        'edge_density': edge_density,
        'qr_raw_detected': qr_detected,
        'qr_raw_text': decoded_text,
        'width': w,
        'height': h
    }

def gerar_entidade():
    """Gera uma configuração de filtros aleatória (genes limpos)."""
    return {
        "kernel_size": random.choice([1, 3, 5, 7, 9]),
        "contrast": random.randint(50, 200),
        "bright": random.randint(50, 200),
        "sat": random.randint(50, 200),
        "sharp": random.randint(0, 100),
        "clahe": random.randint(0, 20),
        "thresh_block": random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),
        "thresh_c": random.randint(0, 10)
    }

def carregar_dataset(dataset_path=DATASET_PATH, silencioso=False):
    """Carrega as imagens do dataset para a memória RAM."""
    if not silencioso:
        print(f"\n⏳ Carregando imagens do dataset para a RAM...")
    imagens = []
    if not os.path.exists(dataset_path):
        if not silencioso:
            print(f"❌ Erro: Pasta '{dataset_path}' não existe.")
        return []

    for nome in os.listdir(dataset_path):
        caminho = os.path.join(dataset_path, nome)
        img = cv2.imread(caminho)
        if img is not None:
            imagens.append((nome, img))

    if not silencioso:
        print(f"✅ {len(imagens)} imagens carregadas com sucesso.")
    return imagens

def salvar_caracteristicas_dataset(dataset):
    """Mapeia e persiste as características físicas de todas as imagens."""
    conn = conectar()
    c = conn.cursor()
    for nome, img in dataset:
        try:
            feat = calcular_caracteristicas_imagem(img)
            c.execute('''INSERT OR IGNORE INTO image_features
                         (image_name, brightness, contrast, saturation, laplacian_variance, edge_density, qr_raw_detected, qr_raw_text, width, height)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (nome, feat['brightness'], feat['contrast'], feat['saturation'], feat['laplacian_variance'], feat['edge_density'],
                       feat['qr_raw_detected'], feat['qr_raw_text'], feat['width'], feat['height']))
        except Exception as e:
            print(f"⚠️ Falha ao extrair características de {nome}: {e}")
    conn.commit()
    conn.close()

def inicializar_worker(seed, dataset_path):
    """Inicializador de cada processo secundário da Pool de processamento."""
    global GLOBAL_DATASET
    cv2.setNumThreads(1) 
    cv2.ocl.setUseOpenCL(False) 
    GLOBAL_DATASET = carregar_dataset(dataset_path, silencioso=True)
    
    if seed is not None:
        worker_seed = seed + os.getpid()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

def avaliar_entidade(args):
    """Aplica filtros e mede o score de sucesso de um indivíduo em todo o dataset."""
    global GLOBAL_DATASET
    entidade, idx = args
    acertos = 0
    total = len(GLOBAL_DATASET)
    dados_para_db = []

    for nome, img_original in GLOBAL_DATASET:
        # Aplica filtros alinhados (GPU com sat/sharp implementado)
        img_proc = aplicar_filtros(img_original, entidade, usar_cuda=USAR_CUDA)
        
        # Tenta decodificar o QR Code processado ignorando fallbacks redundantes para acelerar o GA
        decoded_info, points = decode_qr_image(img_proc, fast_mode=True)

        if decoded_info:
            acertos += 1
            bbox_str = json.dumps(points.tolist()) if points is not None else None
            dados_para_db.append((nome, json.dumps(entidade), 1.0, decoded_info, bbox_str))
        else:
            # Fallback secundário instantâneo via cache de características estáticas
            feat = obter_caracteristicas_com_cache(nome, img_original)
            if feat.get('qr_raw_detected'):
                acertos += 1
                dados_para_db.append((nome, json.dumps(entidade), 0.5, feat.get('qr_raw_text'), None))
            else:
                # Falha completa
                dados_para_db.append((nome, json.dumps(entidade), 0.0, None, None))

    score = acertos / total if total > 0 else 0.0
    return score, entidade, idx, dados_para_db

def crossover(a, b):
    return {k: random.choice([a[k], b[k]]) for k in a}

def mutar(ent):
    """Mutação adaptativa baseada nos genes limpos de filtros."""
    if random.random() < TAXA_MUT:
        k = random.choice(list(ent.keys()))
        
        if random.random() < 0.2: 
            nova_ent = gerar_entidade()
            ent[k] = nova_ent[k]
        else:
            val = ent[k] + random.randint(-40, 40)
            
            limites = {
                "sharp": (0, 500), "clahe": (0, 50),
                "bright": (0, 300), "contrast": (0, 300),
                "sat": (0, 300), "thresh_c": (0, 20)
            }
            
            if k in limites:
                ent[k] = max(limites[k][0], min(val, limites[k][1]))
            elif k == "kernel_size":
                new_k = max(3, min(val, 9))
                ent[k] = new_k + 1 if new_k % 2 == 0 else new_k
            elif k == "thresh_block":
                new_b = max(5, min(val, 51))
                ent[k] = new_b + 1 if new_b % 2 == 0 else new_b
                
    return ent

def torneio(populacao_com_scores):
    candidatos = random.sample(populacao_com_scores, 3)
    candidatos.sort(key=lambda x: x[0], reverse=True)
    return candidatos[0][1]

FEAT_CACHE = {}

def obter_caracteristicas_com_cache(nome, img):
    """Retorna características físicas da imagem lendo do cache ou DB (zero custo de processamento)."""
    global FEAT_CACHE
    if nome in FEAT_CACHE:
        return FEAT_CACHE[nome]
    
    conn = conectar()
    c = conn.cursor()
    c.execute('''SELECT brightness, contrast, saturation, laplacian_variance, edge_density, 
                        qr_raw_detected, qr_raw_text, width, height 
                 FROM image_features WHERE image_name = ?''', (nome,))
    row = c.fetchone()
    conn.close()
    
    if row:
        feat = {
            'brightness': row[0],
            'contrast': row[1],
            'saturation': row[2],
            'laplacian_variance': row[3],
            'edge_density': row[4],
            'qr_raw_detected': row[5],
            'qr_raw_text': row[6],
            'width': row[7],
            'height': row[8]
        }
    else:
        feat = calcular_caracteristicas_imagem(img)
        
    FEAT_CACHE[nome] = feat
    return feat

def sugerir_filtros_pela_rede(dataset, populacao, top_n=3):
    """Usa a MLP treinada como função de fitness substituta para prever o score global."""
    # OTIMIZAÇÃO: Cria 100 candidatos aleatórios + população atual
    piscina_candidatos = populacao + [gerar_entidade() for _ in range(100)]
    
    # Pré-carrega as características de todas as 32 imagens
    todas_features = []
    for nome, img in dataset:
        todas_features.append(obter_caracteristicas_com_cache(nome, img))
        
    from src.ml_model import carregar_modelo, IMAGE_KEYS, FILTER_KEYS
    model = carregar_modelo()
    
    if model is None:
        # Fallback de segurança se a IA não estiver treinada
        return populacao[:top_n]
        
    import numpy as np
    candidatos_scores = []
    
    # A IA atua como um Oráculo: prevê a nota média GLOBAL de cada um dos 100 filtros
    for cand in piscina_candidatos:
        X_test = []
        fvals = [cand.get(k, 0) for k in FILTER_KEYS]
        for feat in todas_features:
            base = [feat.get(k, 0) for k in IMAGE_KEYS]
            X_test.append(base + fvals)
            
        try:
            preds = model.predict(np.array(X_test, dtype=float))
            # O score previsto para este candidato é a média do score em todas as imagens
            score_global = sum(preds) / len(preds)
        except Exception:
            score_global = 0.0
            
        candidatos_scores.append((cand, score_global))
        
    # Ordena os candidatos baseados no maior score global previsto
    candidatos_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retorna apenas os TOP N filtros mais robustos (generalistas) sugeridos pela IA
    melhores_globais = [c for c, s in candidatos_scores[:top_n]]
    
    # Retorna no formato de lista de tamanho igual à população para compatibilidade
    return (melhores_globais + populacao)[:len(populacao)]

def algoritmo_genetico(use_nn=True, pop_size=10, gen_count=10, progress_callback=None):
    """
    Executa o Algoritmo Genético de Otimização de Filtros de Imagem.
    Retorna o resumo da execução e salva o histórico no banco de dados.
    """
    inicializar_db()
    
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)

    dataset_main = carregar_dataset(DATASET_PATH)
    if not dataset_main:
        print("❌ Dataset vazio. Certifique-se de ter imagens sob Dataset/QRCode_diaADia")
        return None

    # Garante que as características físicas do dataset estejam salvas no DB
    salvar_caracteristicas_dataset(dataset_main)

    # OTIMIZAÇÃO: Pré-carrega características em memória para evitar acessos concorrentes ao SQLite
    global FEAT_CACHE
    FEAT_CACHE = {}
    try:
        conn = conectar()
        c = conn.cursor()
        c.execute('''SELECT image_name, brightness, contrast, saturation, laplacian_variance, edge_density, 
                            qr_raw_detected, qr_raw_text, width, height 
                     FROM image_features''')
        for r in c.fetchall():
            FEAT_CACHE[r[0]] = {
                'brightness': r[1],
                'contrast': r[2],
                'saturation': r[3],
                'laplacian_variance': r[4],
                'edge_density': r[5],
                'qr_raw_detected': r[6],
                'qr_raw_text': r[7],
                'width': r[8],
                'height': r[9]
            }
        conn.close()
    except Exception as e:
        print(f"⚠️ Erro ao pré-carregar cache: {e}")

    # População Inicial
    populacao = [gerar_entidade() for _ in range(pop_size)]

    # Se ativado, usa a IA MLP para filtrar e ordenar a população inicial
    if use_nn:
        populacao = sugerir_filtros_pela_rede(dataset_main, populacao)

    start_run_time = time.time()
    historico_geracoes = []
    melhor_score_final = 0.0
    media_score_final = 0.0
    vencedor = None
    
    # Executa de forma sequencial (extremamente estável, leve e veloz com CUDA no Windows)
    global GLOBAL_DATASET
    GLOBAL_DATASET = dataset_main

    for g in range(1, gen_count + 1):
        args = [(ind, i) for i, ind in enumerate(populacao)]
        import sys
        from concurrent.futures import ThreadPoolExecutor
        
        num_workers = min(len(args), os.cpu_count() or 4)
        print(f"   ⏳ [Geração {g:02d}/{gen_count}] Avaliando {pop_size} indivíduos em paralelo ({num_workers} threads)...")
        sys.stdout.flush()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            resultados = list(executor.map(avaliar_entidade, args))

        # Persiste os resultados de extração da geração corrente no DB
        conn_db = conectar()
        c_db = conn_db.cursor()
        for res in resultados:
            _, _, _, dados_i = res
            c_db.executemany('INSERT INTO qr_extractions (image_name, filtros, score, decoded_text, bbox) VALUES (?, ?, ?, ?, ?)', dados_i)
        conn_db.commit()
        conn_db.close()

        # Ordena população baseada nos scores medidos
        avaliacoes = sorted([(r[0], r[1], r[2]) for r in resultados], key=lambda x: x[0], reverse=True)
        melhor_score = avaliacoes[0][0]
        media_score = sum([x[0] for x in avaliacoes]) / len(avaliacoes)

        dados_geracao = {
            'geracao': g,
            'melhor_score': round(melhor_score, 4),
            'media_score': round(media_score, 4),
            'melhor_config': avaliacoes[0][1]
        }
        historico_geracoes.append(dados_geracao)

        # Dispara o callback de progresso para a API / WebSockets
        if progress_callback:
            progress_callback(g, gen_count, melhor_score, media_score, avaliacoes[0][1])

        # Evolução para a próxima geração
        if g < gen_count:
            # 1. Crossover e Mutação Pura (Elitismo)
            nova_pop = [ind for _, ind, _ in avaliacoes[:ELITE]]
            while len(nova_pop) < pop_size:
                pai = torneio(avaliacoes)
                mae = torneio(avaliacoes)
                filho = mutar(crossover(pai, mae))
                nova_pop.append(filho)
            
            # 2. Se ativado, injeta 20% de indivíduos altamente promissores sugeridos pela IA (MLP)
            if use_nn:
                sugestoes = sugerir_filtros_pela_rede(dataset_main, [a for _, a, _ in avaliacoes], top_n=1)
                num_sugestoes = max(1, int(pop_size * 0.2)) # Substitui os piores 20%
                for idx_sug in range(num_sugestoes):
                    if idx_sug < len(sugestoes):
                        nova_pop[-(idx_sug + 1)] = sugestoes[idx_sug]
                        
            populacao = nova_pop

        melhor_score_final = melhor_score
        media_score_final = media_score
        vencedor = avaliacoes[0][1]

    tempo_total = time.time() - start_run_time

    # Salva estatísticas deste experimento no SQLite
    salvar_experimento(
        usar_rede_neural=use_nn,
        geracoes=gen_count,
        populacao=pop_size,
        tempo_total=tempo_total,
        melhor_score=melhor_score_final,
        media_score=media_score_final,
        historico_geracoes=historico_geracoes
    )

    return {
        'usar_rede_neural': use_nn,
        'tempo_total': round(tempo_total, 2),
        'melhor_score_final': melhor_score_final,
        'media_score_final': media_score_final,
        'vencedor': vencedor,
        'historico': historico_geracoes
    }
