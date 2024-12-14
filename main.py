import cv2
from ultralytics import YOLO

#   iniciar camara
cap = cv2.VideoCapture(0)
#   ajustar tamaño
cap.set(3, 640)
cap.set(4, 480)

#   bucle para mostrar la camara
while True:
    #   capturar la camara
    succes, img = cap.read()
    #   mostrar la camara en pantalla
    cv2.imshow('Webcam', img)
    #   definimos una tecla para cerrar la camara
    if cv2.waitKey(1) == ord('q'):
        break

#   liberar la camara
cap.release()
#   cerrar ventana
cv2.destroyAllWindows()