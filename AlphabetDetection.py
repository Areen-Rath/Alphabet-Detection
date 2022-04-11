import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

x = np.load('image.npz')["arr_0"]
y = pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

x_trained, x_test, y_trained, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
x_trained_scaled = x_trained/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_trained_scaled, y_trained)

y_predict = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        pil = Image.fromarray(roi)
        img_bw = pil.convert("L")
        img_bw_resize = img_bw.resize((28, 28), Image.ANTIALIAS)
        invert = PIL.ImageOps.invert(img_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(invert, pixel_filter)
        scaled = np.clip(invert - min_pixel, 0, 255)
        max_pixel = np.max(invert)
        scaled = np.asarray(scaled)/max_pixel

        sample = np.array(scaled).reshape(1, 784)
        test_predict = clf.predict(sample)
        print(test_predict)

        cv2.imshow("Frame", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()