import cv2
import numpy as np
import os
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WECHAT_DIR = os.path.join(BASE_DIR, 'models', 'wechat')

detect_prototxt = os.path.join(WECHAT_DIR, 'detect.prototxt')
detect_caffemodel = os.path.join(WECHAT_DIR, 'detect.caffemodel')
sr_prototxt = os.path.join(WECHAT_DIR, 'sr.prototxt')
sr_caffemodel = os.path.join(WECHAT_DIR, 'sr.caffemodel')

_local_detectors = threading.local()

def inicializar_wechat():
    """Tenta carregar o modelo de Deep Learning do WeChat QR Code de forma thread-safe."""
    global _local_detectors
    if not hasattr(_local_detectors, 'wechat'):
        try:
            if (os.path.exists(detect_prototxt) and os.path.exists(detect_caffemodel) and
                os.path.exists(sr_prototxt) and os.path.exists(sr_caffemodel)):
                
                # Inicializa WeChat QR Code Detector específico para esta thread
                _local_detectors.wechat = cv2.wechat_qrcode_WeChatQRCode(
                    detect_prototxt, detect_caffemodel,
                    sr_prototxt, sr_caffemodel
                )
            else:
                _local_detectors.wechat = None
        except Exception:
            _local_detectors.wechat = None
            
    return _local_detectors.wechat

def decode_qr_image(img, fast_mode=True):
    """
    Pipeline de decodificação unificado e resiliente.
    Se fast_mode=True, ignora fallbacks lentos e redundantes para acelerar o Algoritmo Genético.
    """
    # Garante que a imagem esteja em 3 canais ou escala de cinza
    if img is None:
        return None, None

    # OTIMIZAÇÃO 1: Evita processar imagens sólidas sem informação (brancas ou pretas)
    # O desvio padrão (std) de uma imagem de cor sólida é 0.
    try:
        if np.std(img) < 5.0:
            return None, None
    except Exception:
        pass
        
    # Método 1: WeChat QR Code Detector (Deep Learning)
    wechat = inicializar_wechat()
    if wechat is not None:
        try:
            # detectAndDecode retorna uma tupla (lista_de_textos, lista_de_pontos)
            res = wechat.detectAndDecode(img)
            if res and len(res) == 2:
                texts, points = res
                if texts and len(texts) > 0 and texts[0]:
                    txt = texts[0]
                    # Garante decodificação de bytes em string caso necessário
                    if isinstance(txt, bytes):
                        txt = txt.decode('utf-8', errors='ignore')
                    
                    pts = points[0] if points and len(points) > 0 else None
                    return txt, pts
        except Exception:
            pass

    if fast_mode:
        return None, None

    # Método 2: OpenCV Standard QRCodeDetector
    detector = cv2.QRCodeDetector()
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

    # Fallback com imagens pré-processadas no detector padrão
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    tentativas = [
        gray,
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.bitwise_not(gray)
    ]
    
    for attempt in tentativas:
        try:
            result = detector.detectAndDecode(attempt)
            if isinstance(result, tuple):
                dig = result[0] if len(result) > 0 else None
                pts = result[1] if len(result) > 1 else None
                if dig:
                    return dig, pts
        except Exception:
            continue

    # Tenta rotações de 90, 180 e 270 graus em caso de angulação do QR code (Apenas com detector leve OpenCV)
    for angle in [90, 180, 270]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Tenta detector padrão com imagem rotacionada
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
