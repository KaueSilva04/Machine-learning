import os
import cv2
import numpy as np
import base64
import json
import queue
import threading
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.database import obter_estatisticas_gerais, obter_comparativo_experimentos
from src.filters import aplicar_filtros
from src.decoders import decode_qr_image
from src.ga_engine import algoritmo_genetico

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, 'web')

app = FastAPI(title="Otimização Inteligente de QR Codes")

# Monta arquivos estáticos (CSS, JS, etc.)
static_path = os.path.join(WEB_DIR, 'static')
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def get_index():
    """Serve a página principal HTML."""
    index_path = os.path.join(WEB_DIR, 'templates', 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Página index.html não encontrada. Crie o arquivo templates/index.html"}

@app.get("/api/stats")
async def get_stats():
    """Retorna estatísticas consolidadas do banco de dados."""
    try:
        stats = obter_estatisticas_gerais()
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/comparativo")
async def get_comparativo():
    """Retorna dados de performance comparativa entre as rodadas com e sem IA."""
    try:
        comparativo = obter_comparativo_experimentos()
        return comparativo
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/clear-db")
async def clear_database():
    """Limpa as tabelas de extração e deleta o arquivo de treinamento da IA."""
    try:
        import sqlite3
        conn = sqlite3.connect('data/qrcode_data.db')
        c = conn.cursor()
        c.execute('DELETE FROM qr_extractions')
        c.execute('DELETE FROM ga_experiments')
        conn.commit()
        conn.close()
        
        caminho_modelo = 'models/filtro_predictor.pkl'
        if os.path.exists(caminho_modelo):
            os.remove(caminho_modelo)
            
        return {"status": "success", "message": "Banco de dados e Inteligência Artificial zerados com sucesso!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/test-filter")
async def test_filter(
    file: UploadFile = File(...),
    kernel_size: int = Form(1),
    contrast: int = Form(100),
    bright: int = Form(100),
    sat: int = Form(100),
    sharp: int = Form(0),
    clahe: int = Form(0),
    thresh_block: int = Form(11),
    thresh_c: int = Form(2),
    thresh_type: int = Form(2),
    thresh_val: int = Form(127),
    usar_wechat: bool = Form(True),
    resize_images: bool = Form(True)
):
    """
    Sandbox Interativa. Recebe uma imagem, aplica os parâmetros de
    filtros fornecidos e retorna a imagem binarizada processada e a decodificação.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "message": "Falha ao decodificar a imagem enviada."}
            
        if resize_images:
            h, w = img.shape[:2]
            limite_max = 1600
            if max(h, w) > limite_max:
                escala = limite_max / max(h, w)
                nova_largura = int(w * escala)
                nova_altura = int(h * escala)
                img = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

        cfg = {
            "kernel_size": kernel_size,
            "contrast": contrast,
            "bright": bright,
            "sat": sat,
            "sharp": sharp,
            "clahe": clahe,
            "thresh_block": thresh_block,
            "thresh_c": thresh_c,
            "thresh_type": thresh_type,
            "thresh_val": thresh_val
        }
        
        # Processa imagem (GPU preferencial se CUDA ativo)
        img_proc = aplicar_filtros(img, cfg, usar_cuda=True)
        
        # Tenta decodificar usando decodificador unificado (WeChat + Fallbacks)
        decoded_text, pts = decode_qr_image(img_proc, usar_wechat=usar_wechat)
        
        has_text = bool(decoded_text and decoded_text.strip())
        has_square = pts is not None and len(pts) > 0
        
        # Se encontrou o quadrado (com ou sem texto), desenha ele na imagem!
        if has_square:
            try:
                pts_arr = np.int32(pts).reshape(-1, 2)
                cv2.polylines(img_proc, [pts_arr], True, (0, 255, 0), 3)
            except Exception:
                pass
                
        # Converte imagem final processada em base64 PNG
        _, encoded_img = cv2.imencode(".png", img_proc)
        base64_img = base64.b64encode(encoded_img).decode("utf-8")
        
        if has_text:
            final_status = decoded_text
            success_flag = True
        elif has_square:
            final_status = "Quadrado (Bounding Box) Detectado, mas falhou em ler o texto."
            success_flag = True  # Mostrará checkmark verde para indicar que achou algo
        else:
            final_status = "Não detectado"
            success_flag = False
        
        return {
            "success": True,
            "has_decoded": success_flag,
            "decoded_text": final_status,
            "image": f"data:image/png;base64,{base64_img}"
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/api/auto-filter")
async def auto_filter(
    file: UploadFile = File(...)
):
    """
    Botão Mágico (Oráculo): Recebe uma imagem, gera configurações aleatórias,
    pergunta para a Rede Neural qual é a melhor, e retorna essa configuração.
    """
    try:
        from src.ga_engine import calcular_caracteristicas_imagem, gerar_entidade
        from src.filters import FILTER_KEYS
        from src.ml_model import carregar_modelo, IMAGE_KEYS
        import random

        # Lendo a imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "message": "Imagem inválida."}
            
        model = carregar_modelo()
        if model is None:
            return {"success": False, "message": "Rede Neural não treinada. Rode o algoritmo genético primeiro."}
            
        # Extrai os 8 metadados na ordem exata
        img_features = calcular_caracteristicas_imagem(img)
        img_vals = [img_features[k] for k in IMAGE_KEYS]
        
        # Gera 5000 filtros e faz a predição em BATCH (extremamente rápido)
        configs = []
        X_batch = []
        
        for _ in range(5000):
            cfg = gerar_entidade()
            configs.append(cfg)
            filter_vals = [cfg[k] for k in FILTER_KEYS]
            X_batch.append(img_vals + filter_vals)
            
        X_batch = np.array(X_batch, dtype=float)
        scores = model.predict(X_batch)
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_cfg = configs[best_idx]
                
        return {
            "success": True,
            "predicted_score": best_score,
            "best_filter": best_cfg
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}

@app.get("/api/run-ga")
async def run_ga(use_nn: bool = True, pop: int = 10, gen: int = 10, usar_wechat: bool = True, batch_size: int = 50, resize_images: bool = True):
    """
    Executa o Algoritmo Genético em background de forma assíncrona
    e transmite o progresso da evolução em tempo real via Server-Sent Events (SSE).
    """
    import sys
    print(f"\n🧬 [Painel Web] Iniciando Algoritmo Genético...")
    print(f"   🔹 População: {pop} | Gerações: {gen} | Lote: {batch_size} | Usar IA: {use_nn} | Usar WeChat: {usar_wechat} | Resize: {resize_images}")
    sys.stdout.flush()
    
    event_queue = queue.Queue()
 
    def progress_callback(geracao, total_geracoes, melhor_score, media_score, melhor_cfg):
        # Imprime no terminal do servidor em tempo real
        print(f"   🧬 [Web GA - Geração {geracao:02d}/{total_geracoes}] 🏆 Melhor Score: {melhor_score:.2%} | 📈 Média: {media_score:.2%}")
        sys.stdout.flush()
        
        event_queue.put({
            "type": "progress",
            "generation": geracao,
            "total_generations": total_geracoes,
            "best_score": melhor_score,
            "avg_score": media_score,
            "best_config": melhor_cfg
        })
 
    def run_engine_thread():
        try:
            res = algoritmo_genetico(
                use_nn=use_nn,
                pop_size=pop,
                gen_count=gen,
                usar_wechat=usar_wechat,
                progress_callback=progress_callback,
                batch_size=batch_size,
                resize_images=resize_images
            )
            print(f"✅ [Painel Web] Algoritmo Concluído com Sucesso!")
            sys.stdout.flush()
            event_queue.put({
                "type": "done",
                "result": res
            })
        except Exception as e:
            print(f"❌ [Painel Web] Erro durante o GA: {e}")
            sys.stdout.flush()
            event_queue.put({
                "type": "error",
                "message": str(e)
            })

    # Dispara thread em background
    threading.Thread(target=run_engine_thread, daemon=True).start()

    async def sse_generator():
        done = False
        while not done:
            # Consome itens da fila de eventos em background de forma não-bloqueante
            while not event_queue.empty():
                item = event_queue.get()
                if item["type"] == "error":
                    yield f"data: {json.dumps({'error': item['message']})}\n\n"
                    done = True
                    break
                elif item["type"] == "done":
                    yield f"data: {json.dumps({'done': True, 'result': item['result']})}\n\n"
                    done = True
                    break
                else:
                    yield f"data: {json.dumps(item)}\n\n"
            
            await asyncio.sleep(0.5)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
