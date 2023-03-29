import cv2
import pickle
import csv
import numpy as np

# PARAMETERS
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.80  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRANNIED MODEL
pickle_in = open("models/model_trained.p", "rb")
model = pickle.load(pickle_in)

# STORE THE CLASS NAMES IN A LIST
classNames = []
# read the csv file
with open("data/labels.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # skip header row
    for row in csv_reader:
        className = row[1]
        classNames.append(className)


# PRE-PROCESS THE IMAGES BEFORE FEEDING INTO THE MODEL
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # Convert to grayscale
    img = equalize(img)  # Equalize the histogram
    img = img / 255  # Normalize the image
    return img


def getClassNames(classNo):
    return classNames[int(classNo)]


# PREDICT THE TRAFFIC SIGNS
while True:
    # Read image
    success, imgOrignal = cap.read()

    # Pre-process the image
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Predict
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassNames(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
