import face_recognition
import cv2
import os
import time
from datetime import datetime, timedelta
import shutil
from scipy.spatial import distance as dist


# metodo che si occupa di calcolare i valori dell'occhio
def get_eye(eye):
    # this functon calculate the aspect ratio of the given eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    eye = (A + B) / (C * 2.0)
    # return the eye aspect ratio
    return eye


# metodo che si occupa di controllare se viene eseguito un occhiolino destro o sinistro
def wink_check(res_image):
    # recupero la mappatura del viso
    face_landmarks_list = face_recognition.face_landmarks(res_image)
    for face_landmark in face_landmarks_list:
        # recupero le posizioni degli occhi
        occhiosx = face_landmark["left_eye"]
        occhiodx = face_landmark["right_eye"]
        # calcolo i valori dell'occhio su ogni frame
        apertura_sx = get_eye(occhiosx)
        apertura_dx = get_eye(occhiodx)
        # eseguo la comparazione per individuare la presenza di un occchiolino
        # occhiolino sinistro (speculare)
        if (apertura_sx >= 0.24) and (apertura_dx < 0.23):
            return 1
        # occhiolino destro (speculare)
        elif (apertura_sx < 0.23) and (apertura_dx >= 0.24):
            return 2
        else:
            return 0


# metodo che flagga come true se non sono ancora passati i secondi specificati
# saved_image_name: nome di una stringa che rappresenta la data da controllare
# lapse_length: secondi specificati
def time_flag_seconds(saved_image_name, lapse_length):
    now = datetime.now()
    now_less_seconds = now - timedelta(seconds=lapse_length)
    saved_image_date = datetime.strptime(saved_image_name, "%d_%m_%Y_%H_%M_%S")
    if now_less_seconds >= saved_image_date:
        return False
    return True


# metodo che si occupa di preparare la stringa per la conversione a data rimuovendo estensione e percorso directory
def time_string_prep(the_string, folder):
    the_proper_string = the_string.replace(f"dataset/{folder}/", "")
    the_proper_string = the_proper_string.replace(f"dataset\\{folder}/", "")
    the_proper_string = the_proper_string.replace(".jpg", "")
    return the_proper_string

# cattura della webcam
webcam_capture = cv2.VideoCapture(0)

# dichiarazione variabili
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
        # Tupla che comprende il nome della cartella, encoding dell'immagine, immagine e nome dell'immagine
        print(directory)
        encoding_tup = (directory, target_encoding, target_image, target_image_name)
        list_encoding_tup.append(encoding_tup)


# while che si occupa di gestire le immagini catturate dalla webcam
process_this_frame = True

while True:
    # diminuzione del numero dei frame catturati
    ret, frame = webcam_capture.read()
    # rescale del frame a 1/5 su ogni asse
    small_frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
    # conversione del colore a bgr a rgb
    rgb_small_frame = cv2.cvtColor(small_frame, 4)

    # processa il frame e trasforma l'immagine in una matrice
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        frame_encodings = face_recognition.face_encodings(rgb_small_frame)

        # se è presente un viso
        if frame_encodings:
            frame_face_encoding = frame_encodings[0]

            # se find_a_person è True, allora esegue una ricerca sulle facce presenti nel dataset
            if find_a_person:
                find_a_person = False
                # ciclo for che si occupa di controllare se è presente un match tra l'encoding di
                # ogni tupla e l'encoding del frame catturato
                for image_tup in list_encoding_tup:
                    match = face_recognition.compare_faces([image_tup[1]], frame_face_encoding)[0]

                    # se è presente un match, assegno alla prima label il nome della cartella e il nome dell'immagine
                    # alla seconda label e fermo la ricerca
                    if match:
                        target_name = image_tup[0]
                        image_name = image_tup[3]
                        break
                    # altrimenti assegno alla label "Sconosciuto"
                    else:
                        target_name = "Sconosciuto"

                # sostituisco gli spazi ai lower case e visualizzo a schermo
                shown_name = target_name.replace('_', ' ')
                label = f"{shown_name}"

            # se find_a_person è False significa che è stato individuato un viso ed entra in questo blocco
            else:
                shown_name = target_name.replace('_', ' ')
                label = f"{shown_name}"

                # Il programma ricononosce un utente REGISTRATO
                if match & (target_name != "sconosciuti") & ('visitatore' not in target_name):
                    # se presente, recupera la password dell'utente
                    # 1 = left wink; 2 = right wink
                    isExist = os.path.exists(f"passwords/{target_name}.txt")
                    if isExist:
                        myfile = open(f"passwords/{target_name}.txt", "rt")
                        userpass = myfile.read()
                        myfile.close()
                        # se la password non è completa entra in questo blocco
                        if len(passcheck) < len(userpass):
                            wink = wink_check(rgb_small_frame)
                            # visualizzazione a schermo del numero di occhiolini inseriti sul numero totale
                            cv2.putText(frame, f"{len(passcheck)}/{len(userpass)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                        (255, 0, 0), 2)
                            # controllo che non permette di inserire un occhiolino entro 2 secondi dal precedente
                            if (not time_flag_seconds(time_for_winks, 2)) & (wink != 0):
                                if wink == 1:
                                    # assegnamento alla label
                                    which_eye_is_winking = "left"
                                elif wink == 2:
                                    # assegnamento alla label
                                    which_eye_is_winking = "right"
                                else:
                                    # assegnamento alla label (mai utilizzato, ma evita il warning di possibile
                                    # variabile undefined)
                                    which_eye_is_winking = "not"
                                # aggiornamento della stringa password col valore corrispondente
                                passcheck = passcheck + f"{wink_check(rgb_small_frame)}"
                                # avverte l'utente a schermo dell'occhiolino effettuato
                                label = f"{shown_name} is {which_eye_is_winking} winking."
                                # aggiorna il timer dell'occhiolino
                                time_for_winks = time.strftime("%d_%m_%Y_%H_%M_%S")
                        # se la password è completa entra in questo blocco
                        else:
                            # se la password è corretta entra in questo blocco
                            if passcheck == userpass:
                                label = f"Password corretta per {target_name}"
                                print("Cancello aperto")
                            # se la password è errata entra in questo blocco
                            else:
                                label = f"Password errata per {target_name}"
                            # Reset della password inserita
                            passcheck = ""
                            # Riattivo la ricerca di persona
                            find_a_person = True

                # Caso in cui la webcam riconosce un utente VISITATORE
                # Un visitatore è un utente che è stato registrato più volte dal sistema
                elif match & (target_name != "sconosciuti") & ('visitatore' in target_name):
                    # Se c'è un'immagine fatta entro sessanta secondi, non ne salva un'altra
                    chrono_trigger = False
                    # Ciclo che si occupa di controllare gli orari dei frame salvati
                    for (i, item) in enumerate(list_encoding_tup):
                        if "visitatore" in item[0]:
                            date_string = time_string_prep(item[3], item[0])
                            if time_flag_seconds(date_string, 60):
                                chrono_trigger = True
                                break
                    # Se è passato abbastanza tempo entra in questo blocco
                    if not chrono_trigger:
                        # Salvataggio del frame
                        snapshot = time.strftime("%d_%m_%Y_%H_%M_%S")
                        cv2.imwrite(f"dataset/{target_name}/{snapshot}.jpg", frame)
                        # Aggiunta alla lista delle tuple
                        new_name = f"dataset/{target_name}/{snapshot}.jpg"
                        target_image = face_recognition.load_image_file(new_name)
                        target_encoding = face_recognition.face_encodings(target_image)[0]
                        encoding_tup = (f"{target_name}", target_encoding, target_image, new_name)
                        list_encoding_tup.append(encoding_tup)
                        # Riattivo la ricerca di persona
                        find_a_person = True

                # Caso in cui la cam riconosce un utente SCONOSCIUTO
                # Uno sconosciuto è un utente che è stato registrato una sola volta dal sistema
                elif match & (target_name == "sconosciuti"):
                    # Se l'immagine è stata catturata entro trenta secondi, non ne salva un'altra, altrimenti
                    # entra nel blocco
                    image_name_prep = time_string_prep(image_name, "sconosciuti")
                    if not time_flag_seconds(image_name_prep, 30):
                        # Creazione nuova cartella visitatoreX
                        counter = 0
                        isExist = os.path.exists(f"dataset/visitatore{counter}")
                        # Controllo del nome delle cartelle per crearne di nuove con numero sequenziale
                        while isExist:
                            counter = counter + 1
                            isExist = os.path.exists(f"dataset/visitatore{counter}")
                        new_directory = f"dataset/visitatore{counter}"
                        os.mkdir(new_directory)
                        # Salvataggio frame E spostamento immagine da sconosciuti a visitatoreX
                        for (i, item) in enumerate(list_encoding_tup):
                            if item[3] == image_name:
                                # sostituisco la stringa sconosciuti con visitatoreX
                                old_name = item[3]
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
                                # creazione tupla del frame e inserimento nella lista di tuple
                                target_image = face_recognition.load_image_file(new_photo_new_name)
                                target_encoding = face_recognition.face_encodings(target_image)[0]
                                encoding_tup = (f"visitatore{counter}", target_encoding, target_image,
                                                new_photo_new_name)
                                list_encoding_tup.append(encoding_tup)
                    # Riattivo la ricerca di persona
                    find_a_person = True

                # Caso in cui la cam riconosce un nuovo sconosciuto
                # Un nuovo sconosciuto è una persona che viene rilevata per la prima volta dal programma
                else:
                    chrono_trigger = False
                    # Ciclo che si occupa di controllare che non sia stato registrato un altro sconosciuto
                    # entro 10 secondi
                    for (i, item) in enumerate(list_encoding_tup):
                        if "sconosciuti" in item[0]:
                            date_string = time_string_prep(item[3], item[0])
                            if time_flag_seconds(date_string, 10):
                                chrono_trigger = True
                                break
                    # Se sono passati almeno 10 secondi dall'ultimo frame salvato, entra nel blocco
                    if not chrono_trigger:
                        # Salvataggio del frame nella cartella sconosciuti
                        snapshot = time.strftime("%d_%m_%Y_%H_%M_%S")
                        cv2.imwrite(f"dataset/sconosciuti/{snapshot}.jpg", frame)
                        # Aggiunta del frame alla lista delle tuple
                        new_name = f"dataset/sconosciuti/{snapshot}.jpg"
                        target_image = face_recognition.load_image_file(new_name)
                        target_encoding = face_recognition.face_encodings(target_image)[0]
                        encoding_tup = ("sconosciuti", target_encoding, target_image, new_name)
                        list_encoding_tup.append(encoding_tup)
                        # Riattivo la ricerca di persona
                        find_a_person = True

        # Se il programma non rileva una faccia entra nel blocco
        if not face_locations:
            # Riattivo la ricerca di persona
            find_a_person = True
            # Resetto l'eventuale password inserita
            passcheck = ""

        # Se il programma rileva una faccia entro nel blocco
        if face_locations:
            # Assegno le posizioni marginali del viso alle variabili
            top, right, bottom, left = face_locations[0]
            top = top * 5
            right *= 5
            bottom *= 5
            left *= 5

            # Creazione del rettangolo per circondare il viso riconosciuto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)

            # Mostro a schermo la label di fianco al rettangolo
            label_font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, label, (left + 6, bottom - 6), label_font, 0.8, (255, 255, 255), 1)

    # Mostro l'immagine a schermo
    cv2.imshow("Video Feed", frame)

    # Esco dal ciclo premendo il tasto Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Chiudo il programma
webcam_capture.release()
cv2.destroyAllWindows()
