
# This is slow








import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("v1.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh()

while True:
    success, img = cap.read()
    if not success:
        break  # Exit if video ends or fails to read frame

    # Resize the image for faster processing
    img = cv2.resize(img, (640, 480))  # Adjust resolution as needed

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Corrected cv2.putText parameters
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

cap.release()
cv2.destroyAllWindows()






