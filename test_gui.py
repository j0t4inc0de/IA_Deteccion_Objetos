#librerías necesarias para la ejecución del programa
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

#creamos la clase DetectorObjetos
class DetectorObjetos(QWidget):
    #inicializamos la clase DetectorObjetos con el constructor
    def __init__(self):
        super().__init__()
        #llamamos al constructor de la clase padre
        self.initUI()
        #inicializamos el modelo YOLO
        self.model = YOLO("best.pt")
        #creamos una lista con los objetos que queremos detectar
        self.objetos = ['dog']
        self.translations = {'dog':'perro'}
        print(self.model.names)
        #inicializamos el timer para la camara y la variable camera_active 
        self.timer = QTimer(self)
        #conectamos el timer con la funcion update_frame
        self.timer.timeout.connect(self.update_frame)
        #inicializamos la variable camera_active en False 
        self.camera_active = False
    #inicializamos la interfaz de usuario
    def initUI(self):
        self.setWindowTitle('Detector de Objetos')
        #definimos el tamaño de la ventana
        self.setGeometry(100, 100, 800, 600)
        #creamos un layout vertical
        layout = QVBoxLayout()
        #creamos un label para mostrar la imagen
        self.image_label = QLabel(self)
        #centramos el label en el layout
        self.image_label.setAlignment(Qt.AlignCenter)
        #agregamos el label al layout
        layout.addWidget(self.image_label)
        #creamos un label para mostrar la cantidad de objetos detectados
        self.count_label = QLabel('Objetos detectados: 0', self)
        #centramos el label en el layout
        layout.addWidget(self.count_label)
        #creamos un layout horizontal para los botones
        btn_layout = QHBoxLayout()
        #creamos un botón para cargar una imagen
        self.upload_btn = QPushButton('Subir Imagen', self)
        #conectamos el botón con la funcion upload_image
        self.upload_btn.clicked.connect(self.upload_image)
        #agregamos el botón al layout
        btn_layout.addWidget(self.upload_btn)
        #creamos un botón para abrir la camara
        self.camera_btn = QPushButton('Abrir Camara', self)
        #conectamos el botón con la funcion toggle_camera
        self.camera_btn.clicked.connect(self.toggle_camera)
        #agregamos el botón al layout
        btn_layout.addWidget(self.camera_btn)
        #agregamos el layout de los botones al layout principal
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    #definimos la función para cargar una imagen
    def upload_image(self):
        #abrimos un cuadro de dialogo para seleccionar una imagen
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", "Images (*.png *.jpg *.bmp)")
        #si se selecciona una imagen
        if file_name:
            #leemos la imagen con opencv
            image = cv2.imread(file_name)
            #procesamos la imagen con el modelo YOLO y la mostramos en pantalla
            self.process_image(image)
            
    #definimos la función para abrir la camara
    def toggle_camera(self):
        #si la camara no esta activa la activamos
        if not self.camera_active:
            #inicializamos la camara
            self.cap = cv2.VideoCapture(0)
            #iniciamos el timer para capturar los frames de la camara
            self.timer.start(30)
            #cambiamos el estado de la camara a activo
            self.camera_active = True
            #cambiamos el texto del botón a detener camara
            self.camera_btn.setText('Detener')
        else:
            #si la camara esta activa la detenemos
            self.timer.stop()
            #liberamos la camara
            self.cap.release()
            #cambiamos el estado de la camara a inactivo
            self.camera_active = False
            self.camera_btn.setText('Abrir Camera')
            self.image_label.clear()

    #definimos la función para capturar los frames de la camara
    def update_frame(self):
        #leemos un frame de la camara
        ret, frame = self.cap.read()
        #si el frame se lee correctamente lo procesamos
        if ret:
            #procesamos el frame con el modelo YOLO
            self.process_image(frame)
            
    #definimos la función para procesar una imagen
    def process_image(self, image):
        #procesamos la imagen con el modelo YOLO
        resultados = self.model(image, stream=True)
        #contado de objetos detectados
        count = 0
        #recorremos los resultados de la detección
        for r in resultados:
            #obtenemos las cajas de detección
            boxes = r.boxes
            #recorremos las cajas de detección
            for box in boxes:
                #obtenemos la clase del objeto detectado
                cls = int(box.cls[0])
                #obtenemos el nombre de la clase
                class_name = self.model.names[cls]
                #si la clase del objeto detectado esta en la lista de objetos
                if class_name in self.objetos:
                    #obtenemos la confianza de la detección
                    confidence = float(box.conf[0])
                    #si la confianza es mayor al 70%
                    if confidence > 0.7:
                        #incrementamos el contador de objetos detectados
                        count += 1
                        #obtenemos las coordenadas de la caja de detección
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        #dibujamos la caja de detección en la imagen
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        #escribimos el nombre de la clase y la confianza de la detección en la imagen
                        spanish_name = self.translations.get(class_name, class_name)
                        cv2.putText(image, f"{spanish_name} {confidence*100:.1f}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        #actualizamos el label con la cantidad de objetos detectados
        self.count_label.setText(f'Objetos detectados: {count}')
        self.display_image(image)
        
    #definimos la función para mostrar una imagen en el label
    def display_image(self, img):
        #convertimos la imagen de BGR a RGB
        qformat = QImage.Format_RGB888
        #convertimos la imagen de opencv a QImage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #obtenemos las dimensiones de la imagen
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        #mostramos la imagen en el label
        self.image_label.setPixmap(QPixmap.fromImage(outImage))
        #ajustamos el tamaño del label a la imagen
        self.image_label.setScaledContents(True)



#inicio de la aplicación
app = QApplication(sys.argv)
ex = DetectorObjetos()
ex.show()
sys.exit(app.exec_())