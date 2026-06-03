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
USAR_CUDA = True
SEED = 42

GLOBAL_DATASET = []
GLOBAL_DATASET_PATH = DATASET_PATH

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
        "bright": random.randint(50, 150),
        "sharp": random.randint(0, 100),
        "clahe": random.randint(0, 50),
        "thresh_type": random.randint(0, 2),
        "thresh_val": random.randint(50, 200),
        "thresh_block": random.choice([x for x in range(3, 55) if x % 2 != 0]),
        "thresh_c": random.randint(0, 10)
    }

def carregar_dataset(dataset_path=DATASET_PATH, silencioso=False, resize_images=True):
    """Carrega as imagens do dataset para a memória RAM com redimensionamento opcional."""
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
            if resize_images:
                h, w = img.shape[:2]
                limite_max = 1600
                if max(h, w) > limite_max:
                    escala = limite_max / max(h, w)
                    nova_largura = int(w * escala)
                    nova_altura = int(h * escala)
                    img = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
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
    global GLOBAL_DATASET, GLOBAL_DATASET_PATH
    cv2.setNumThreads(1) 
    cv2.ocl.setUseOpenCL(False) 
    GLOBAL_DATASET_PATH = dataset_path
    
    # Guarda apenas os nomes dos arquivos em vez de carregar gigabytes de imagens na RAM por processo
    if os.path.exists(dataset_path):
        GLOBAL_DATASET = [nome for nome in os.listdir(dataset_path) if nome.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        GLOBAL_DATASET = []
    
    if seed is not None:
        worker_seed = seed + os.getpid()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

def avaliar_entidade(args):
    """Aplica filtros e mede o score de sucesso de um indivíduo em todo o dataset."""
    batch_names = None
    resize_images = True
    if len(args) == 5:
        entidade, idx, usar_wechat, batch_names, resize_images = args
    elif len(args) == 4:
        entidade, idx, usar_wechat, batch_names = args
    elif len(args) == 3:
        entidade, idx, usar_wechat = args
    else:
        entidade, idx = args
        usar_wechat = True
        
    acertos = 0
    
    global GLOBAL_DATASET_PATH, GLOBAL_DATASET
    dataset_to_use = batch_names if batch_names is not None else GLOBAL_DATASET
    total = len(dataset_to_use)
    
    dados_para_db = []

    for item in dataset_to_use:
        if isinstance(item, tuple):
            nome, img_original = item
        else:
            nome = item
            img_original = None
            
        if img_original is None:
            caminho = os.path.join(GLOBAL_DATASET_PATH, nome)
            img_original = cv2.imread(caminho)
            if img_original is not None and resize_images:
                h, w = img_original.shape[:2]
                limite_max = 1600
                if max(h, w) > limite_max:
                    escala = limite_max / max(h, w)
                    nova_largura = int(w * escala)
                    nova_altura = int(h * escala)
                    img_original = cv2.resize(img_original, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
            
        if img_original is None:
            dados_para_db.append((nome, json.dumps(entidade), 0.0, None, None))
            continue
            
        # Aplica filtros (GPU com sat/sharp implementado)
        img_proc = aplicar_filtros(img_original, entidade, usar_cuda=USAR_CUDA)
        
        # Tenta decodificar a imagem processada
        decoded_info, points = decode_qr_image(img_proc, usar_wechat=usar_wechat, fast_mode=True)

        # O ÚNICO OBJETIVO: Achar o quadro do QR Code
        if points is not None and len(points) > 0:
            acertos += 1
            bbox_str = json.dumps(points.tolist()) if hasattr(points, 'tolist') else str(points)
            info_salva = decoded_info if decoded_info else ""
            dados_para_db.append((nome, json.dumps(entidade), 1.0, info_salva, bbox_str))
        else:
            # Falha total do filtro na imagem
            dados_para_db.append((nome, json.dumps(entidade), 0.0, None, None))

    # O Score final é puramente a porcentagem de quadros encontrados no dataset inteiro!
    score = acertos / total if total > 0 else 0.0
    return score, entidade, idx, dados_para_db

def crossover(a, b):
    return {k: random.choice([a[k], b[k]]) for k in a}

def mutar(ent):
    """Mutação agressiva: Avalia CADA gene independentemente para forçar diversidade e quebrar estagnação."""
    for k in list(ent.keys()):
        # 30% de chance de mutar CADA parâmetro individualmente (pode mutar vários ao mesmo tempo!)
        if random.random() < 0.30:
            if random.random() < 0.15: 
                nova_ent = gerar_entidade()
                ent[k] = nova_ent[k]
            else:
                val = ent[k] + random.randint(-50, 50)
                
                limites = {
                    "sharp": (0, 500), "clahe": (0, 50),
                    "bright": (0, 300), "contrast": (0, 300),
                    "thresh_c": (0, 20), "thresh_val": (0, 255)
                }
                
                if k in limites:
                    ent[k] = max(limites[k][0], min(val, limites[k][1]))
                elif k == "kernel_size":
                    new_k = max(3, min(val, 15))
                    ent[k] = new_k + 1 if new_k % 2 == 0 else new_k
                elif k == "thresh_block":
                    new_b = max(3, min(val, 51))
                    ent[k] = new_b + 1 if new_b % 2 == 0 else new_b
                elif k == "thresh_type":
                    ent[k] = random.randint(0, 2)
                    
    # Fallback para garantir campos necessários se não existirem
    if 'sharp' not in ent: ent['sharp'] = 0
    if 'clahe' not in ent: ent['clahe'] = 0
    if 'thresh_type' not in ent: ent['thresh_type'] = 2
    if 'thresh_val' not in ent: ent['thresh_val'] = 127
    if 'thresh_block' not in ent: ent['thresh_block'] = 11

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
    # OTIMIZAÇÃO: Cria 1000 candidatos baseados em MUTAÇÕES DA ELITE + população atual
    melhor_elite = populacao[0]
    piscina_candidatos = populacao + [mutar(dict(melhor_elite)) for _ in range(1000)]
    
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
    
    # A IA atua como um Oráculo: OTIMIZAÇÃO BATCH (Matricial)
    # Em vez de chamar a IA 1000 vezes, criamos uma mega matriz e chamamos 1 vez só!
    X_batch = []
    num_imgs = len(todas_features)
    
    if num_imgs > 0:
        for cand in piscina_candidatos:
            fvals = [cand.get(k, 0) for k in FILTER_KEYS]
            for feat in todas_features:
                base = [feat.get(k, 0) for k in IMAGE_KEYS]
                X_batch.append(base + fvals)
                
        try:
            # Disparo único para a Rede Neural
            todas_preds = model.predict(np.array(X_batch, dtype=float))
            
            for i, cand in enumerate(piscina_candidatos):
                preds_do_cand = todas_preds[i * num_imgs : (i + 1) * num_imgs]
                score_global = sum(preds_do_cand) / num_imgs
                candidatos_scores.append((cand, score_global))
        except Exception:
            for cand in piscina_candidatos:
                candidatos_scores.append((cand, 0.0))
    else:
        for cand in piscina_candidatos:
            candidatos_scores.append((cand, 0.0))
        
    # Ordena os candidatos baseados no maior score global previsto
    candidatos_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retorna apenas os TOP N filtros mais robustos (generalistas) sugeridos pela IA
    melhores_globais = [c for c, s in candidatos_scores[:top_n]]
    
    # Retorna no formato de lista de tamanho igual à população para compatibilidade
    return (melhores_globais + populacao)[:len(populacao)]

def mutacao_guiada_por_ia(filho_original, todas_features, model, n_opcoes=5):
    """
    Cria n_opcoes de mutações a partir do filho. 
    Usa a IA para prever o score global de cada mutação e retorna a melhor.
    """
    from src.ml_model import IMAGE_KEYS, FILTER_KEYS
    import numpy as np
    
    opcoes = []
    # Cria clones mutados
    for _ in range(n_opcoes):
        clone = dict(filho_original)
        opcoes.append(mutar(clone))
        
    candidatos_scores = []
    X_batch = []
    num_imgs = len(todas_features)
    
    if num_imgs > 0:
        for cand in opcoes:
            fvals = [cand.get(k, 0) for k in FILTER_KEYS]
            for feat in todas_features:
                base = [feat.get(k, 0) for k in IMAGE_KEYS]
                X_batch.append(base + fvals)
                
        try:
            # Predição matricial super rápida
            todas_preds = model.predict(np.array(X_batch, dtype=float))
            
            for i, cand in enumerate(opcoes):
                preds_do_cand = todas_preds[i * num_imgs : (i + 1) * num_imgs]
                score_global = sum(preds_do_cand) / num_imgs
                candidatos_scores.append((cand, score_global))
        except Exception:
            for cand in opcoes:
                candidatos_scores.append((cand, 0.0))
    else:
        for cand in opcoes:
            candidatos_scores.append((cand, 0.0))
            
    candidatos_scores.sort(key=lambda x: x[1], reverse=True)
    return candidatos_scores[0][0] # Retorna o filtro mutado que obteve o maior score previsto

def algoritmo_genetico(use_nn=True, pop_size=10, gen_count=10, usar_wechat=True, progress_callback=None, batch_size=0, resize_images=True, batch_fixed=True):
    """
    Executa o Algoritmo Genético de Otimização de Filtros de Imagem.
    Retorna o resumo da execução e salva o histórico no banco de dados.
    """
    inicializar_db()
    
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)

    dataset_main = carregar_dataset(DATASET_PATH, resize_images=resize_images)
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
    
    # Executa de forma sequencial ou paralela dependendo do CUDA
    cuda_disponivel = False
    if USAR_CUDA:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cuda_disponivel = True
        except Exception:
            pass

    global GLOBAL_DATASET
    GLOBAL_DATASET = dataset_main

    # Se não houver CUDA, cria a Pool de processos uma única vez ANTES do loop das gerações (Persistent Pool)
    pool = None
    if not cuda_disponivel:
        from multiprocessing import Pool
        num_workers = min(pop_size, os.cpu_count() or 4)
        print(f"   ⏳ [Inicialização] Criando Pool de {num_workers} processos de CPU...")
        sys.stdout.flush()
        pool = Pool(processes=num_workers, initializer=inicializar_worker, initargs=(SEED, DATASET_PATH))

    # --- LOTE FIXO (Opcional) ---
    global_fixed_batch = None
    if batch_fixed and batch_size > 0 and batch_size < len(dataset_main):
        global_fixed_batch = random.sample(dataset_main, batch_size)
        if not cuda_disponivel:
            global_fixed_batch = [item[0] if isinstance(item, tuple) else item for item in global_fixed_batch]

    try:
        for g in range(1, gen_count + 1):
            if global_fixed_batch is not None:
                batch_to_use = global_fixed_batch
            elif batch_size > 0 and batch_size < len(dataset_main):
                batch_to_use = random.sample(dataset_main, batch_size)
                if not cuda_disponivel:
                    batch_to_use = [item[0] if isinstance(item, tuple) else item for item in batch_to_use]
            else:
                batch_to_use = None
                
            args = [(ind, i, usar_wechat, batch_to_use, resize_images) for i, ind in enumerate(populacao)]
            import sys
            import gc
            
            resultados = []

            if cuda_disponivel:
                print(f"   ⏳ [Geração {g:02d}/{gen_count}] Avaliando {pop_size} indivíduos em modo sequencial ultra-rápido com GPU/CUDA (RAM pré-carregada)...")
                sys.stdout.flush()
                # Rodar sequencialmente na GPU aproveitando o cache da RAM
                for arg in args:
                    resultados.append(avaliar_entidade(arg))
            else:
                print(f"   ⏳ [Geração {g:02d}/{gen_count}] Avaliando {pop_size} indivíduos em paralelo com Pool persistente...")
                sys.stdout.flush()
                resultados = pool.map(avaliar_entidade, args)
                
            # Força o Python a limpar os vestígios pesados de imagem da memória RAM
            gc.collect()

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
            
            # LOG PEDIDO PELO USUÁRIO: Mostra o placar detalhado da geração
            print(f"   📊 [Placar da Geração {g:02d}/{gen_count}]")
            for score_ind, ind_config, idx_ind in avaliacoes:
                # Mostra o score e também a configuração exata que o Algoritmo testou
                config_str = ", ".join([f"{k}: {v}" for k, v in ind_config.items()])
                print(f"      🔹 Filtro {idx_ind:02d}: {(score_ind*100):.1f}% | Config: [{config_str}]")

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
                # 1. Crossover e Mutação (Guiada pela IA ou Cega)
                nova_pop = [ind for _, ind, _ in avaliacoes[:ELITE]]
                
                # Prepara o modelo da IA para guiar as mutações
                ia_model = None
                todas_features = []
                if use_nn:
                    from src.ml_model import carregar_modelo
                    ia_model = carregar_modelo()
                    if ia_model is not None:
                        for nome, img in dataset_main:
                            todas_features.append(obter_caracteristicas_com_cache(nome, img))
                            
                while len(nova_pop) < pop_size:
                    pai = torneio(avaliacoes)
                    mae = torneio(avaliacoes)
                    filho = crossover(pai, mae)
                    
                    # MUTAÇÃO GUIADA PELA IA (Impede regressões evolutivas)
                    if use_nn and ia_model is not None:
                        filho = mutacao_guiada_por_ia(filho, todas_features, ia_model, n_opcoes=5)
                    else:
                        filho = mutar(filho)
                        
                    nova_pop.append(filho)
                
                # 2. Se ativado, injeta indivíduos altamente promissores sugeridos pela IA (Macro-Evolução)
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
    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
