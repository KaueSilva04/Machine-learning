import cv2
import numpy as np

# Apenas os parâmetros reais que influenciam ativamente o processamento
FILTER_KEYS = ['kernel_size', 'contrast', 'bright', 'sat', 'sharp', 'clahe', 'thresh_block', 'thresh_c']

def aplicar_filtros_cpu(img_original, cfg):
    """Aplica o pipeline de filtros de imagem em CPU."""
    img = img_original.copy()
    try:
        # 1. Median Blur (Suavização)
        ksize = cfg.get("kernel_size", 1)
        if ksize % 2 == 0:
            ksize += 1
        if ksize > 1:
            img = cv2.medianBlur(img, ksize)
            
        # 2. Contraste e Brilho
        if cfg.get("contrast", 100) != 100 or cfg.get("bright", 100) != 100:
            alpha = cfg["contrast"] / 100.0
            beta = cfg["bright"] - 100
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 3. Saturação (Espaço de Cor HSV)
        if cfg.get("sat", 100) != 100:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (cfg["sat"] / 100.0), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 4. Nitidez (Sharpening via filtro convolucional 2D)
        if cfg.get("sharp", 0) > 0:
            sharp_factor = cfg["sharp"] / 100.0
            kernel = np.array([
                [0, -1, 0],
                [-1, sharp_factor + 4, -1],
                [0, -1, 0]
            ])
            img = cv2.filter2D(img, -1, kernel)

        # 5. Conversão para tons de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 6. CLAHE (Equalização de contraste adaptativa)
        if cfg.get("clahe", 0) > 0:
            clahe = cv2.createCLAHE(clipLimit=cfg["clahe"]/10.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
        # 7. Limiarização Adaptativa (Adaptive Thresholding)
        block_size = cfg.get("thresh_block", 11)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
            
        return cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            cfg.get("thresh_c", 2)
        )
    except Exception:
        # Fallback de segurança para escala de cinza pura
        return cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

def aplicar_filtros_cuda(img_original, cfg):
    """Aplica o pipeline de filtros de imagem em GPU utilizando OpenCV CUDA."""
    try:
        img_cpu = img_original.copy()
        
        # 1. Median Blur (Operado em CPU antes de subir, pois CUDA não tem implementação direta nativa simples de medianBlur)
        ksize = cfg.get("kernel_size", 1)
        if ksize % 2 == 0:
            ksize += 1
        if ksize > 1:
            img_cpu = cv2.medianBlur(img_cpu, ksize)
            
        # Upload para GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_cpu)

        # 2. Contraste e Brilho (GPU)
        if cfg.get("contrast", 100) != 100 or cfg.get("bright", 100) != 100:
            alpha = cfg["contrast"] / 100.0
            beta = cfg["bright"] - 100
            gpu_img = gpu_img.convertTo(-1, alpha=alpha, beta=beta)

        # 3. Saturação (GPU + manipulação do canal S)
        if cfg.get("sat", 100) != 100:
            gpu_hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
            channels = cv2.cuda.split(gpu_hsv)
            
            # Download temporário e veloz apenas do canal S
            s_cpu = channels[1].download()
            s_cpu = np.clip(s_cpu * (cfg["sat"] / 100.0), 0, 255).astype(np.uint8)
            channels[1].upload(s_cpu)
            
            gpu_hsv = cv2.cuda.merge(channels)
            gpu_img = cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2BGR)

        # 4. Nitidez (GPU via filtro linear CUDA 2D)
        if cfg.get("sharp", 0) > 0:
            sharp_factor = cfg["sharp"] / 100.0
            kernel = np.array([
                [0, -1, 0],
                [-1, sharp_factor + 4, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            
            # Filtro linear CUDA 2D
            filter_gpu = cv2.cuda.createLinearFilter(gpu_img.type(), gpu_img.type(), kernel)
            gpu_img = filter_gpu.apply(gpu_img)

        # 5. Tons de cinza (GPU)
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

        # 6. CLAHE (GPU)
        if cfg.get("clahe", 0) > 0:
            clahe_gpu = cv2.cuda.createCLAHE(clipLimit=cfg["clahe"]/10.0, tileGridSize=(8,8))
            stream = cv2.cuda_Stream()
            gpu_gray = clahe_gpu.apply(gpu_gray, stream)

        # Download do resultado final em cinza para a CPU operar a limiarização
        gray_cpu = gpu_gray.download()

        # 7. Limiarização Adaptativa
        block_size = cfg.get("thresh_block", 11)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
            
        return cv2.adaptiveThreshold(
            gray_cpu, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            cfg.get("thresh_c", 2)
        )
    except Exception as e:
        # Em caso de erro na CUDA, faz o fallback gracioso para a CPU
        return aplicar_filtros_cpu(img_original, cfg)

def aplicar_filtros(img_original, cfg, usar_cuda=True):
    """
    Função pública de entrada. Decide se rodará na GPU (CUDA) 
    ou fará o processamento direto em CPU.
    """
    if usar_cuda:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return aplicar_filtros_cuda(img_original, cfg)
        except (AttributeError, Exception):
            pass
            
    return aplicar_filtros_cpu(img_original, cfg)
