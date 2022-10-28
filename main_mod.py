# time test for face recognition and facial expressions (wink and smile)
import face_recognition
import cv2 as cv
# if not in google colab delete or comment the following line from PIL import Image
import time
from scipy.spatial import distance as dist


def main():

    biden_image = face_recognition.load_image_file("photos/biden.jpeg")
    megan_image = face_recognition.load_image_file("photos/megan.jpg")

    try:
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        megan_face_encoding = face_recognition.face_encodings(megan_image)[0]
    except:
        print("I didn't any faces, qutting...\n")
        time.sleep(0.02)
        print(".\n")
        time.sleep(0.02)
        print(".\n")
        time.sleep(0.02)
        print(".\n")

    known_face_encodings = [
        biden_face_encoding,
        megan_face_encoding
    ]

    occhiolino = False
    sorriso = False
    sblocco = [False]
    t_sblocco = 0
    t_occhiolino = 0
    # gets the time
    t_start = time.time()

    # change the string below with the preferred option:
    # "Winki.jpg" "biden.jpeg" "biden2.jpg" "biden3.jpg" "biden4.jpg" "biden5.jpg" "bush.jpg"
    # "megan.jpg" "megan2.jpg" "megan3.jpg" "megan4.jpg" "not_smile.jpg" "wink2.png"
    # note that only the files with "biden" or "megan will be recognized"
    file = "pi1.jpg"

    image = cv.imread(f"photos/{file}")
    # Pitzus - rimossi i magic numbers
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    res_image = cv.resize(image, dim)
    # get encoding from the image of the first face
    print(f"face_recognition {face_recognition.face_encodings(res_image)}")
    new_encoding = face_recognition.face_encodings(res_image)[0]
    # compare with the known faces
    sblocco = face_recognition.compare_faces(known_face_encodings, new_encoding)
    # if there is at least one face that is recognized this will return True
    if True in sblocco:
        t_sblocco = time.time()
    # get the coordinates of the first face
    # Pitz - Variable below was not a list (fixed)
    face_landmarks_list = face_recognition.face_landmarks(res_image)
    for face_landmark in face_landmarks_list:
        # get the eyes
        occhiosx = face_landmark["left_eye"]
        occhiodx = face_landmark["right_eye"]
        # calculate the aspect ratio of the eyes
        apertura_sx = get_eye(occhiosx)
        apertura_dx = get_eye(occhiodx)
        # compare with estimated threshold
        occhiolino = ((apertura_sx >= 0.23) and (apertura_dx < 0.23)) or (
                    (apertura_sx < 0.23) and (apertura_dx >= 0.23))
        # gets the time fot the facial expr. wink
        if occhiolino:
            t_occhiolino = time.time()

    # print of the results
    if True in sblocco:
        print(f"unlocked at:       {t_sblocco - t_start}")
    else:
        print("error, subject not recognized")
    if occhiolino:
        print(f"wink at:                {t_occhiolino - t_start}")
    else:
        print(f"error wink not found")
    print(f"test concluded in:      {t_end - t_start}")
    print(f"image dimensions:       {image.shape}")
    # cv2_imshow(image)  # if not in google colab comment this line and de comment the following
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_eye(eye):
    # this functon calculate the aspect ratio of the given eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    eye = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return eye


if __name__ == "__main__":
    main()