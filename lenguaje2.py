import cv2

# Cargar el clasificador preentrenado para rostros y sonrisas
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Abrir la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterar sobre cada rostro
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detectar sonrisas dentro del rostro
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.8, minNeighbors=20)

        # Determinar la emoción
        if len(smiles) > 0:
            emotion = "Feliz 😊"
            color = (0, 255, 0)
        else:
            emotion = "Triste 😢"
            color = (0, 0, 255)

        # Mostrar la emoción en la imagen
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar la imagen en pantalla
    cv2.imshow("Detector de Emociones (Feliz o Triste)", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
