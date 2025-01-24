
#! Instalamos las librerias necesarias

# pip install ultralytics
# pip install roboflow

#! Esperamos que de descarguen e iniciamos
import torch
print('Estado de CUDA:', torch.cuda.is_available())
print('Version:', torch.version.cuda)
print('Numero de dispositivos:', torch.cuda.device_count())
print('Dispositivos:', torch.cuda.get_device_name(0))

from roboflow import Roboflow
rf = Roboflow(api_key="JHs0I92NKOAPCY1PuJsw")
project = rf.workspace("mooreungdetection").project("mooreung_detection")
version = project.version(1)
dataset = version.download("yolov8")

#! Configuramos la ruta de los datasets
# configurar ruta del datasets
data_yaml = '/content/Mooreung_detection-1/data.yaml'
from ultralytics import YOLO
# crear modelo YOLO
model = YOLO('yolov8n.pt')

#! Iniciamos el entrenamiento
resultado = model.train(
    data = data_yaml,
    epochs = 100,
    imgsz = 640,
    batch = 16,
    device = 'cuda',
    project = 'entrenamiento',
    name = 'yolo_custom'
)
validar = model.val()

model.export(format='onnx')

#! Con esto podemos impimir las clases que tiene el modelo
model = YOLO('/content/entrenamiento/yolo_custom/weights/best.pt')
print(model.names)