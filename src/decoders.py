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

def decode_qr_image(img, fast_mode=True, usar_wechat=True):
    """
    Pipeline de decodificação unificado e resiliente.
    Se usar_wechat=True, tenta o modelo com IA do WeChat.
    Se usar_wechat=False ou falhar, usa o detector nativo/rápido do OpenCV.
    Se fast_mode=True, ignora fallbacks lentos de rotação.
    """
    if img is None:
        return None, None

    # OTIMIZAÇÃO 1: Evita processar imagens sólidas sem informação (brancas ou pretas)
    try:
        _, stddev = cv2.meanStdDev(img)
        if stddev.mean() < 5.0:
            return None, None
    except Exception:
        pass

    # 1. WeChat QR Code Detector (IA / Deep Learning)
    if usar_wechat:
        wechat = inicializar_wechat()
        if wechat is not None:
            try:
                res = wechat.detectAndDecode(img)
                if res and len(res) == 2:
                    texts, points = res
                    if points and len(points) > 0 and len(points[0]) > 0:
                        pts = points[0]
                        txt = texts[0] if texts and len(texts) > 0 else ""
                        if isinstance(txt, bytes):
                            txt = txt.decode('utf-8', errors='ignore')
                        return txt, pts
            except Exception:
                pass

        if fast_mode:
            return None, None

    # 2. OpenCV Standard QRCodeDetector (Nativo e Rápido)
    detector = cv2.QRCodeDetector()
    try:
        if fast_mode:
            # Se for fast_mode (utilizado no AG para buscar o quadrado), usamos apenas o detect() que é ultra-rápido
            retval, points = detector.detect(img)
            if retval and points is not None and len(points) > 0:
                if len(points.shape) == 3:
                    points = points[0]
                return "", points
            return None, None
        else:
            # Caso contrário, tenta decodificar por completo
            result = detector.detectAndDecode(img)
            if isinstance(result, tuple):
                if len(result) == 3:
                    decoded, points, _ = result
                elif len(result) == 4:
                    _, decoded, points, _ = result
            if decoded:
                if points is not None and len(points.shape) == 3:
                    points = points[0]
                return decoded, points
    except Exception:
        pass

    if fast_mode:
        return None, None

    # Fallbacks mais lentos para decodificação completa (somente se fast_mode=False)
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
                    if pts is not None and len(pts.shape) == 3:
                        pts = pts[0]
                    return dig, pts
        except Exception:
            continue

    # Tenta rotações de 90, 180 e 270 graus em caso de angulação do QR code (Apenas com detector leve OpenCV)
    for angle in [90, 180, 270]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        try:
            result = detector.detectAndDecode(rotated)
            if isinstance(result, tuple):
                dig = result[0] if len(result) > 0 else None
                pts = result[1] if len(result) > 1 else None
                if dig:
                    if pts is not None and len(pts.shape) == 3:
                        pts = pts[0]
                    return dig, pts
        except Exception:
            continue

    return None, None
