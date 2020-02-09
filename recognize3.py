import cv2
time1 = 0
import time 
power = 0
#float displaytime = 0 
color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors, 0, (125,125))
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        # Check for id of user and label the rectangle accordingly
    

        if id==1:
            cv2.putText(img, "Kevin", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            cv2.putText(img, "Kevin is present", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

        if id != 1:
            cv2.putText(img, "Unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            cv2.putText(img, "_____ is present", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 20, color["white"], "Face", clf)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    

    # Call method we defined above
    img = recognize(img, clf, faceCascade)
    # Writing processed image in a new window
    
    
    time1 += 3
    displaytime = "{:12.2f}".format(time1/60)
    img = cv2.putText(img, str(displaytime), (-100, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow("face detection", img)
    if time1 >= 181.5:
        time1 = 0
        img = recognize(img, clf, faceCascade)
        img = cv2.putText(img, "Updated", (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 10, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("face detection", img)
        time.sleep(2)
        power = 1
        break 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while power == 1:
    _, img = video_capture.read()
    

    # Call method we defined above
    img = recognize(img, clf, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()