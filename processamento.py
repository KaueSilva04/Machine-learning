import os
os.environ["OPENCV_OPENCL_CACHE_ENABLE"] = "false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import time
import sqlite3
import json

from ml_sugerir_filtro import carregar_modelo, sugerir_filtros, FILTER_KEYS, IMAGE_KEYS

DATASET_PATH = "./Dataset/QRCode_diaADia"
POP = 10  # Reduzido para teste rápido
GERACOES = 10  # Executar 10 gerações conforme solicitado
TAXA_MUT = 0.7 
ELITE = 5
USAR_CUDA = True 
NUM_PROCESSOS = max(1, cpu_count() - 1)
SEED = 42 

GLOBAL_DATASET = []

def inicializar_db():
    conn = sqlite3.connect('qrcode_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS qr_extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        filtros TEXT,
        score REAL,
        decoded_text TEXT,
        bbox TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
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
    conn.commit()
    conn.close()

def calcular_caracteristicas_imagem(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:,:,1]))

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / (w * h))

    detector = cv2.QRCodeDetector()
    decoded_text = None
    qr_detected = 0
    try:
        raw = detector.detectAndDecode(img)
        if isinstance(raw, tuple):
            if len(raw) == 3:
                decoded_text = raw[0]
            elif len(raw) == 4:
                decoded_text = raw[1]
        if decoded_text:
            qr_detected = 1
    except Exception:
        qr_detected = 0

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
    return {
        "kernel_size": random.choice([1, 3, 5, 7, 9]),
        "contrast": random.randint(50, 200),
        "bright": random.randint(50, 200),
        "sat": random.randint(50, 200),
        "sharp": random.randint(0, 100),
        "clahe": random.randint(0, 20),
        "thresh_block": random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),
        "thresh_c": random.randint(0, 10),
        "gamma": random.randint(50, 150),
        "denoise": random.randint(0, 10),
        "expo": random.randint(50, 150)
    }

def aplicar_filtros_cuda(img_original, cfg):
    try:
        img_cpu = img_original.copy()
        ksize = cfg["kernel_size"]
        if ksize % 2 == 0: ksize += 1
        if ksize > 1:
            img_cpu = cv2.medianBlur(img_cpu, ksize)

        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_cpu)

        if cfg["contrast"] != 100 or cfg["bright"] != 100:
            alpha = cfg["contrast"] / 100.0
            beta  = cfg["bright"] - 100
            gpu_img = gpu_img.convertTo(-1, alpha=alpha, beta=beta)

        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

        if cfg["clahe"] > 0:
            clahe_gpu = cv2.cuda.createCLAHE(clipLimit=cfg["clahe"]/10.0, tileGridSize=(8,8))
            stream = cv2.cuda_Stream()
            gpu_gray = clahe_gpu.apply(gpu_gray, stream)

        gray_cpu = gpu_gray.download()

        block_size = cfg["thresh_block"]
        if block_size % 2 == 0: block_size += 1
        if block_size < 3: block_size = 3
        
        final_img = cv2.adaptiveThreshold(gray_cpu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, cfg["thresh_c"])
        return final_img

    except Exception as e:
        print(f"⚠️ ERRO CUDA: {e}")
        return aplicar_filtros_cpu(img_original, cfg)

def aplicar_filtros_cpu(img_original, cfg):
    img = img_original.copy()
    try:
        ksize = cfg["kernel_size"]
        if ksize % 2 == 0: ksize += 1
        if ksize > 1: img = cv2.medianBlur(img, ksize)
            
        if cfg["contrast"] != 100 or cfg["bright"] != 100:
            alpha, beta = cfg["contrast"] / 100.0, cfg["bright"] - 100
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if cfg["sat"] != 100:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (cfg["sat"] / 100.0), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if cfg["sharp"] > 0:
            sharp_factor = cfg["sharp"] / 100.0
            kernel = np.array([[0, -1, 0], [-1, sharp_factor + 4, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cfg["clahe"] > 0:
            clahe = cv2.createCLAHE(clipLimit=cfg["clahe"]/10.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
        block_size = cfg["thresh_block"]
        if block_size % 2 == 0: block_size += 1
        if block_size < 3: block_size = 3
        
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, cfg["thresh_c"])
    except Exception:
        return cv2.cvtColor(img_original, COLOR_BGR2GRAY)

def aplicar_filtros(img_original, cfg):
    if USAR_CUDA:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return aplicar_filtros_cuda(img_original, cfg)
        except AttributeError:
            pass 
            
    return aplicar_filtros_cpu(img_original, cfg)


def decode_qr_image(img):
    detector = cv2.QRCodeDetector()
    # OpenCV retorna 3 ou 4 valores dependendo da versão
    decoded, points = None, None
    try:
        result = detector.detectAndDecode(img)
        if isinstance(result, tuple):
            if len(result) == 3:
                decoded, points, _ = result
            elif len(result) == 4:
                _, decoded, points, _ = result
        if decoded:
            return decoded, points
    except Exception:
        pass

    # Fallback com imagens pre-processadas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    for attempt in [gray, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), cv2.bitwise_not(gray)]:
        try:
            result = detector.detectAndDecode(attempt)
            if isinstance(result, tuple):
                dig = result[0] if len(result) > 0 else None
                pts = result[1] if len(result) > 1 else None
                if dig:
                    return dig, pts
        except Exception:
            continue

    # Tentar rotações 90, 180, 270
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        try:
            result = detector.detectAndDecode(rotated)
            if isinstance(result, tuple):
                dig = result[0] if len(result) > 0 else None
                pts = result[1] if len(result) > 1 else None
                if dig:
                    return dig, pts
        except Exception:
            continue

    return None, None


def carregar_dataset(dataset_path, silencioso=False):
    if not silencioso:
        print(f"\n⏳ A carregar imagens para a RAM...")
    imagens = []
    if not os.path.exists(dataset_path):
        if not silencioso:
            print(f"❌ Erro: Pasta '{dataset_path}' não existe. Crie a pasta e coloque imagens lá.")
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
    conn = sqlite3.connect('qrcode_data.db')
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
    global GLOBAL_DATASET
    cv2.setNumThreads(1) 
    cv2.ocl.setUseOpenCL(False) 
    
    GLOBAL_DATASET = carregar_dataset(dataset_path, silencioso=True)
    
    if seed is not None:
        worker_seed = seed + os.getpid()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

def avaliar_entidade(args):
    global GLOBAL_DATASET
    entidade, idx = args

    print(f"   [Indivíduo {idx:02d}] ⏳ A iniciar análise...")
    start_time = time.time()

    total = len(GLOBAL_DATASET)
    acertos = 0
    dados_para_db = []

    for nome, img_original in GLOBAL_DATASET:
        img_proc = aplicar_filtros(img_original, entidade)
        decoded_info, points = decode_qr_image(img_proc)

        if decoded_info:
            acertos += 1
            bbox_str = json.dumps(points.tolist()) if points is not None else None
            dados_para_db.append((nome, json.dumps(entidade), 1.0, decoded_info, bbox_str))
        else:
            decoded_info_raw, points_raw = decode_qr_image(img_original)
            if decoded_info_raw:
                acertos += 1
                bbox_str = json.dumps(points_raw.tolist()) if points_raw is not None else None
                dados_para_db.append((nome, json.dumps(entidade), 0.5, decoded_info_raw, bbox_str))
            else:
                dados_para_db.append((nome, json.dumps(entidade), 0.0, None, None))

    score = acertos / total if total > 0 else 0
    tempo_decorrido = time.time() - start_time
    print(f"   [Indivíduo {idx:02d}] ✅ Concluído em {tempo_decorrido:.2f}s | Acertos: {acertos}/{total} ({score:.2%})")

    return score, entidade, idx, dados_para_db

def crossover(a, b):
    return {k: random.choice([a[k], b[k]]) for k in a}

def mutar(ent):
    if random.random() < TAXA_MUT:
        k = random.choice(list(ent.keys()))
        
        if random.random() < 0.2: 
            nova_ent = gerar_entidade()
            ent[k] = nova_ent[k]
        else:
            val = ent[k] + random.randint(-40, 40)
            
            limites = {
                "sharp": (0, 500), "gamma": (50, 300), "clahe": (0, 50),
                "denoise": (0, 30), "bright": (0, 300), "contrast": (0, 300),
                "expo": (0, 300), "sat": (0, 300), "thresh_c": (0, 20)
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

def sugerir_filtros_pela_rede(dataset, populacao, top_n=3):
    model = carregar_modelo()
    if model is None:
        return populacao

    novas = []
    for nome, img in dataset:
        feat_img = calcular_caracteristicas_imagem(img)
        candidatos = populacao
        recomendados = sugerir_filtros(feat_img, candidatos, top_n=top_n)
        for filt, _ in recomendados:
            if filt not in novas:
                novas.append(filt)
    return (novas[:POP] + populacao)[:POP]


def algoritmo_genetico(use_nn=True):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
    
    status_cuda = "ATIVA 🟢" if (USAR_CUDA and cv2.cuda.getCudaEnabledDeviceCount() > 0) else "INATIVA 🔴 (A usar CPU)"
    
    dataset_main = carregar_dataset(DATASET_PATH)
    if not dataset_main:
        return None

    salvar_caracteristicas_dataset(dataset_main)

    populacao = [gerar_entidade() for _ in range(POP)]

    if use_nn:
        populacao = sugerir_filtros_pela_rede(dataset_main, populacao)

    modo = "COM rede neural" if use_nn else "SEM rede neural"
    print(f"\n🚀 A INICIAR ALGORITMO GENÉTICO {modo} (TREINO PARA TELEMÓVEL)")
    print(f"🔧 Motor Gráfico (NVIDIA CUDA): {status_cuda}")
    print(f"🧠 Processos Paralelos: {NUM_PROCESSOS}")
    print(f"==========================================================\n")
    
    geracoes_sem_melhora = 0
    ultimo_melhor_score = 0.0
    historico = []

    with Pool(NUM_PROCESSOS, initializer=inicializar_worker, initargs=(SEED, DATASET_PATH)) as pool:
        for g in range(1, GERACOES + 1):
            print(f"\n" + "="*50)
            print(f"🧬 A INICIAR GERAÇÃO {g:02d}")
            print("="*50)
            
            start_time = time.time()
            args = [(ind, i) for i, ind in enumerate(populacao)]
            resultados = pool.map(avaliar_entidade, args)

            conn_db = sqlite3.connect('qrcode_data.db')
            c_db = conn_db.cursor()
            for res in resultados:
                score_i, ent_i, idx_i, dados_i = res
                c_db.executemany('INSERT INTO qr_extractions (image_name, filtros, score, decoded_text, bbox) VALUES (?, ?, ?, ?, ?)', dados_i)
            conn_db.commit()
            conn_db.close()

            avaliacoes = sorted([(r[0], r[1], r[2]) for r in resultados], key=lambda x: x[0], reverse=True)

            melhor_score = avaliacoes[0][0]
            media_score = sum([x[0] for x in avaliacoes]) / len(avaliacoes)

            historico.append({
                'geracao': g,
                'melhor': melhor_score,
                'media': media_score,
                'estagnacao': geracoes_sem_melhora
            })

            if use_nn and g < GERACOES:
                best_based_rede = sugerir_filtros_pela_rede(dataset_main, [a for _, a, _ in avaliacoes])
                while len(best_based_rede) < POP:
                    best_based_rede.append(gerar_entidade())
                populacao = best_based_rede
            else:
                nova_pop = [ind for _, ind, _ in avaliacoes[:ELITE]]
                if geracoes_sem_melhora >= 6:
                    print("⚠️  APOCALIPSE! O algoritmo ficou preso. A injetar sangue novo totalmente aleatório...")
                    while len(nova_pop) < POP:
                        nova_pop.append(gerar_entidade())
                    geracoes_sem_melhora = 0
                else:
                    while len(nova_pop) < POP:
                        pai = torneio(avaliacoes)
                        mae = torneio(avaliacoes)
                        filho = mutar(crossover(pai, mae))
                        nova_pop.append(filho)
                populacao = nova_pop

            if melhor_score > ultimo_melhor_score:
                ultimo_melhor_score = melhor_score
                geracoes_sem_melhora = 0
            else:
                geracoes_sem_melhora += 1

            tempo_gen = time.time() - start_time
            print(f"\n📊 RESUMO DA GERAÇÃO {g:02d}")
            print(f"⏳ Tempo Total: {tempo_gen:.2f}s | 🏆 Melhor: {melhor_score:.2%} | 📈 Média: {media_score:.2%} | 🛑 Estagnação: {geracoes_sem_melhora}/6")
            print("💡 Melhor Configuração Atual:")
            melhor_cfg_atual = avaliacoes[0][1]
            for gene, valor in melhor_cfg_atual.items():
                print(f"   🔹 {gene}: {valor}")

    vencedor = avaliacoes[0][1]
    print("\n" + "="*50)
    print("🏆 ALGORITMO CONCLUÍDO! CONFIGURAÇÃO VENCEDORA FINAL:")
    print("="*50)
    for gene, valor in vencedor.items():
        print(f"   🔹 {gene}: {valor}")
    print("\n👉 Estes valores estão agora perfeitamente otimizados para melhorar a visibilidade em câmaras de telemóvel!")

    return {
        'modo': modo,
        'historico': historico,
        'melhor_final': melhor_score,
        'media_final': media_score,
        'vencedor': vencedor
    }

if __name__ == "__main__":
    inicializar_db()

    resultados = []
    for usar_nn in [False, True]:
        res = algoritmo_genetico(use_nn=usar_nn)
        if res is not None:
            resultados.append(res)

    if resultados:
        print("\n\n📊 COMPARAÇÃO FINAL (SEM vs COM rede neural)")
        print("Geracao | Modo                | Melhor final | Media final")
        print("--------|---------------------|--------------|-------------")
        for r in resultados:
            print(f"{GERACOES:7d} | {r['modo']:19s} | {r['melhor_final']:.2%}     | {r['media_final']:.2%}")

        melhor_com = next((r for r in resultados if r['modo'] == 'COM rede neural'), None)
        melhor_sem = next((r for r in resultados if r['modo'] == 'SEM rede neural'), None)
        if melhor_com and melhor_sem:
            ganho_melhor = melhor_com['melhor_final'] - melhor_sem['melhor_final']
            ganho_media = melhor_com['media_final'] - melhor_sem['media_final']
            print(f"\n🔍 Resultado: diferença de melhor = {ganho_melhor:.2%}, diferença de media = {ganho_media:.2%}")
