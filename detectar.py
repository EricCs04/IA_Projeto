from ultralytics import YOLO
import cv2
import os
from glob import glob


# Carrega o modelo YOLOv8 pré-treinado (nano = leve)
model = YOLO('yolov8n.pt')

os.makedirs('resultado', exist_ok=True)

# Caminho da imagem
img_paths = glob('imagens/*.jpg')

for img_path in img_paths:
    print (f"processando: {img_path}")
    # Faz a inferência
    results = model(img_path)
    # Exibe os resultados
    results[0].show()  # Abre uma janela com as detecções

    nome_arquivo = os.path.basename(img_path)
    output_path = f'resultado/{nome_arquivo}'
    results[0].save(filename=output_path)

    print(f"Salvo em: {output_path}")


