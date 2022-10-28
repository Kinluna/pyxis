import face_recognition
import cv2
import os

webcam_capture = cv2.VideoCapture(0)

dataset_path = "dataset"
list_dir = os.listdir(dataset_path)

list_encoding_tup = []

# questo ciclo si occupa di riempire una lista di tuple, formate da encoding dell'immagine e rispettiva directory
for directory in list_dir:
    people_path = os.listdir(os.path.join(dataset_path, directory))
    for image in people_path:
        image_path = os.path.join(os.path.join(dataset_path, directory), image)
        print(f"a {image_path}")
        target_image = face_recognition.load_image_file(image_path)
        target_encoding = face_recognition.face_encodings(target_image)[0]
        encoding_tup = (image_path, target_encoding)
        list_encoding_tup.append(encoding_tup)



target_name = ""
# while che si occupa di gestire le immagini catturate dalla webcam
process_this_frame = True

while True:
    ret, frame = webcam_capture.read()
    # rescale del frame a 1/5 su ogni asse
    small_frame = cv2.resize(frame, None, fx=0.20, fy=0.20)
    # conversione del colore a bgr a rgb
    rgb_small_frame = cv2.cvtColor(small_frame, 4)

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        frame_encodings = face_recognition.face_encodings(rgb_small_frame)

        if frame_encodings:
            frame_face_encoding = frame_encodings[0]

            # ciclo for che si occupa di controllare se è presente un match tra l'encoding di ogni tupla e l'encoding del frame catturato
            for image_tup in list_encoding_tup:
                match = face_recognition.compare_faces([image_tup[1]], frame_face_encoding)[0]
                print(image_tup[0])
                # gestione di ciò che viene mostrato a schermo sulla webcam
                target_name = image_tup[0]
                label = f"{target_name}" if match else "unknown"

            # NYI nuovo if che controlla se nel frame è presente il wink; se presente apri cancello


    process_this_frame = not process_this_frame

    if face_locations:
        top, right, bottom, left = face_locations[0]
        top = top * 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)

        label_font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, label, (left + 6, bottom - 6), label_font, 0.8, (0, 0, 0), 1)

    cv2.imshow("Video Feed", frame)

    # Quit with q

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_capture.release()
cv2.destoyAllWindows()