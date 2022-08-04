import cv2
import dlib
from scipy.spatial import distance
def eye_aspect_ratio(eye):
    p2_minus_p6 = distance.euclidean(eye[1], eye[5])
    p3_minus_p5 = distance.euclidean(eye[2], eye[4])
    p1_minus_p4 = distance.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
n1=0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        
        left_eye = []
        right_eye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	left_eye.append((x,y))
        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	right_eye.append((x,y))
        ear1 = eye_aspect_ratio(left_eye)
        ear2 = eye_aspect_ratio(right_eye)
        ear = (ear1+ear2)/2
        if ear < 0.24:
            for n in range(0, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
        else:
            for n in range(0, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
