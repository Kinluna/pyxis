import this
import face_recognition
import cv2
import os
import time
from datetime import datetime, timedelta
import shutil
from scipy.spatial import distance as dist


def get_smile_tl(top_lip):
    # this function check if the borders of the mouth are higher than the center
    return top_lip[0][1] < top_lip[4][1] or \
           top_lip[6][1] < top_lip[2][1] or \
           top_lip[0][1] < top_lip[6][1] or \
           top_lip[6][1] < top_lip[0][1]


def get_smile_op(top_lip, bottom_lip):
    # this function calculate the aspect ratio of the mouth
    # Pitzus - aggiunto check per apertura bocca
    inner_mouth_distance = dist.euclidean(bottom_lip[9], top_lip[9])
    outer_mouth_distance = dist.euclidean(bottom_lip[2], top_lip[2])

    if inner_mouth_distance != 0:
        mouth_distance = inner_mouth_distance
    else:
        mouth_distance = outer_mouth_distance

    sar = dist.euclidean(top_lip[0], top_lip[6]) / mouth_distance
    return 3 < sar < 5.5


def get_eye(eye):
    # this functon calculate the aspect ratio of the given eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    eye = (A + B) / (C * 2.0)
    # return the eye aspect ratio
    return eye


def smile_check(res_image):
    face_landmarks_list = face_recognition.face_landmarks(res_image)
    for face_landmark in face_landmarks_list:
        # gets lips
        toplip = face_landmark["top_lip"]
        bottomlip = face_landmark["bottom_lip"]
        # check if the subject is smiling
        smiletl = get_smile_tl(toplip)
        smileop = get_smile_op(toplip, bottomlip)
        sorriso = smiletl or smileop
        # compare with estimated threshold
        # gets the time fot the facial expr. wink
        if sorriso:
            return True
        return False


def specular_right_wink_check(res_image):
    face_landmarks_list = face_recognition.face_landmarks(res_image)
    for face_landmark in face_landmarks_list:
        # get the eyes
        occhiosx = face_landmark["left_eye"]
        occhiodx = face_landmark["right_eye"]
        # calculate the aspect ratio of the eyes
        apertura_sx = get_eye(occhiosx)
        apertura_dx = get_eye(occhiodx)
        # compare with estimated threshold
        occhiolino = ((apertura_sx < 0.23) and (apertura_dx >= 0.25))
        # gets the time fot the facial expr. wink
        if occhiolino:
            return True
        return False


def specular_left_wink_check(res_image):
    face_landmarks_list = face_recognition.face_landmarks(res_image)
    for face_landmark in face_landmarks_list:
        # get the eyes
        occhiosx = face_landmark["left_eye"]
        occhiodx = face_landmark["right_eye"]
        # calculate the aspect ratio of the eyes
        apertura_sx = get_eye(occhiosx)
        apertura_dx = get_eye(occhiodx)
        # compare with estimated threshold
        occhiolino = ((apertura_sx >= 0.25) and (apertura_dx < 0.23))
        # gets the time fot the facial expr. wink
        if occhiolino:
            return True
        return False


# metodo che flagga come true se non sono ancora passati 5 minuti
def time_flag_visitors(saved_image_name):
    now = datetime.now()
    now_less_five_minutes = now - timedelta(minutes=5)
    saved_image_date = datetime.strptime(saved_image_name, "%d_%m_%Y_%H_%M_%S")
    if now_less_five_minutes >= saved_image_date:
        return False
    return True


# metodo che flagga come true se non sono ancora passati 2 secondi
def time_flag_unknown(saved_image_name):
    now = datetime.now()
    now_less_two_seconds = now - timedelta(seconds=2)
    saved_image_date = datetime.strptime(saved_image_name, "%d_%m_%Y_%H_%M_%S")
    if now_less_two_seconds >= saved_image_date:
        return False
    return True


# metodo che si occupa di preparare la stringa per la conversione a data
def time_string_prep(the_string, folder):
    the_proper_string = the_string.replace(f"dataset/{folder}/", "")
    the_proper_string = the_proper_string.replace(f"dataset\\{folder}/", "")
    the_proper_string = the_proper_string.replace(".jpg", "")
    return the_proper_string


webcam_capture = cv2.VideoCapture(0)

dataset_path = "dataset"
passcheck = ""
target_name = ""
image_name = ""
find_a_person = True
match = False
label = ""
time_for_winks = time.strftime("%d_%m_%Y_%H_%M_%S")

list_dir = os.listdir(dataset_path)
list_encoding_tup = []

# questo ciclo si occupa di riempire una lista di tuple, formate da encoding dell'immagine e rispettiva directory
for directory in list_dir:
    people_path = os.listdir(os.path.join(dataset_path, directory))
    for image in people_path:
        image_path = os.path.join(dataset_path, directory)
        target_image_name = f"{image_path}/{image}"
        target_image = face_recognition.load_image_file(f"{target_image_name}")
        target_encoding = face_recognition.face_encodings(target_image)[0]
        # Tupla che comprende il nome della cartella e l'encoding dell'immagine
        print(directory)
        encoding_tup = (directory, target_encoding, target_image, target_image_name)
        list_encoding_tup.append(encoding_tup)


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

            # ciclo for che si occupa di controllare se ?? presente un match tra l'encoding di
            # ogni tupla e l'encoding del frame catturato
            if find_a_person:
                find_a_person = False
                for image_tup in list_encoding_tup:
                    match = face_recognition.compare_faces([image_tup[1]], frame_face_encoding)[0]
                    if match:
                        target_name = image_tup[0]
                        image_name = image_tup[3]
                        break
                    else:
                        target_name = "Sconosciuto"

                shown_name = target_name.replace('_', ' ')
                label = f"{shown_name}"

            # Caso in cui la cam individua un utente VISITATORE
            else:
                shown_name = target_name.replace('_', ' ')
                label = f"{shown_name}"

                # Il programma ricononosce un utente
                if match & (target_name != "sconosciuti") & ('visitatore' not in target_name):
                    # se riconosce la persona recupera la password corrispondente
                    # 1 = left wink; 2 = right wink; 3 = smile
                    isExist = os.path.exists(f"passwords/{target_name}.txt")
                    if isExist:
                        myfile = open(f"passwords/{target_name}.txt", "rt")
                        userpass = myfile.read()
                        myfile.close()
                        # metodo di controllo
                        if len(passcheck) < len(userpass):
                            cv2.putText(frame, f"{len(passcheck)}/{len(userpass)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                        (255, 0, 0), 2)
                            if (not time_flag_unknown(time_for_winks)) & specular_left_wink_check(rgb_small_frame):
                                passcheck = passcheck + "1"
                                label = f"{shown_name} is left winking."
                                time_for_winks = time.strftime("%d_%m_%Y_%H_%M_%S")
                            elif (not time_flag_unknown(time_for_winks)) & specular_right_wink_check(rgb_small_frame):
                                label = f"{shown_name} is right winking."
                                passcheck = passcheck + "2"
                                time_for_winks = time.strftime("%d_%m_%Y_%H_%M_%S")
                        else:
                            if passcheck == userpass:
                                label = f"Password corretta per {target_name}"
                                print("Cancello aperto")
                            else:
                                label = f"Password errata per {target_name}"
                            passcheck = ""
                            find_a_person = True

                # Caso in cui la webcam riconosce un visitatore
                elif match & (target_name != "sconosciuti") & ('visitatore' in target_name):
                    # Se c'?? un'immagine fatta entro cinque minuti, non ne salva un'altra
                    chrono_trigger = False
                    for (i, item) in enumerate(list_encoding_tup):
                        if "visitatore" in item[0]:
                            date_string = time_string_prep(item[3], item[0])
                            if time_flag_visitors(date_string):
                                chrono_trigger = True
                                break
                    if not chrono_trigger:
                        # salva frame
                        snapshot = time.strftime("%d_%m_%Y_%H_%M_%S")
                        cv2.imwrite(f"dataset/{target_name}/{snapshot}.jpg", frame)
                        # Aggiunta alla lista delle tuple
                        new_name = f"dataset/{target_name}/{snapshot}.jpg"
                        target_image = face_recognition.load_image_file(new_name)
                        target_encoding = face_recognition.face_encodings(target_image)[0]
                        encoding_tup = (f"{target_name}", target_encoding, target_image, new_name)
                        list_encoding_tup.append(encoding_tup)
                        find_a_person = True

                # Caso in cui la cam riconosce uno sconosciuto
                elif match & (target_name == "sconosciuti"):
                    counter = 0
                    isExist = os.path.exists(f"dataset/visitatore{counter}")
                    while isExist:
                        counter = counter + 1
                        isExist = os.path.exists(f"dataset/visitatore{counter}")
                    new_directory = f"dataset/visitatore{counter}"
                    os.mkdir(new_directory)
                    # Salvataggio frame E spostamento immagine da sconosciuti a visitatoreX
                    for (i, item) in enumerate(list_encoding_tup):
                        if item[3] == image_name:
                            # spostamento da folder sconosciuti a visitatoreX
                            old_name = item[3]
                            # sostituisco la stringa sconosciuti con visitatoreX
                            old_photo_new_name = old_name.replace('sconosciuti', f'visitatore{counter}')
                            shutil.move(old_name, old_photo_new_name)
                            # creazione nuova tupla
                            old_photo_new_image = face_recognition.load_image_file(old_photo_new_name)
                            old_photo_new_encoding = face_recognition.face_encodings(old_photo_new_image)[0]
                            old_photo_new_encoding_tup = (f"visitatore{counter}", old_photo_new_encoding,
                                                          old_photo_new_image, old_photo_new_name)
                            list_encoding_tup.append(old_photo_new_encoding_tup)
                            # rimozione vecchia tupla
                            list_encoding_tup.pop(i)
                            # salvataggio frame in visitatoreX
                            snapshot = time.strftime("%d_%m_%Y_%H_%M_%S")
                            new_photo_new_name = f"{new_directory}/{snapshot}.jpg"
                            cv2.imwrite(new_photo_new_name, frame)
                            target_image = face_recognition.load_image_file(new_photo_new_name)
                            target_encoding = face_recognition.face_encodings(target_image)[0]
                            encoding_tup = (f"visitatore{counter}", target_encoding, target_image, new_photo_new_name)
                            list_encoding_tup.append(encoding_tup)
                    find_a_person = True

                # Caso in cui la cam riconosce un nuovo sconosciuto
                else:
                    chrono_trigger = False
                    for (i, item) in enumerate(list_encoding_tup):
                        if "sconosciuti" in item[0]:
                            date_string = time_string_prep(item[3], item[0])
                            if time_flag_visitors(date_string):
                                chrono_trigger = True
                                break
                    if not chrono_trigger:
                        snapshot = time.strftime("%d_%m_%Y_%H_%M_%S")
                        cv2.imwrite(f"dataset/sconosciuti/{snapshot}.jpg", frame)
                        # Aggiunta alla lista delle tuple
                        new_name = f"dataset/sconosciuti/{snapshot}.jpg"
                        target_image = face_recognition.load_image_file(new_name)
                        target_encoding = face_recognition.face_encodings(target_image)[0]
                        encoding_tup = ("sconosciuti", target_encoding, target_image, new_name)
                        list_encoding_tup.append(encoding_tup)
                        find_a_person = True

        if not face_locations:
            find_a_person = True
            passcheck = ""

        if face_locations:
            top, right, bottom, left = face_locations[0]
            top = top * 5
            right *= 5
            bottom *= 5
            left *= 5

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)

            label_font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, label, (left + 6, bottom - 6), label_font, 0.8, (255, 255, 255), 1)

    cv2.imshow("Video Feed", frame)

    # Quit with q

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_capture.release()
cv2.destroyAllWindows()
