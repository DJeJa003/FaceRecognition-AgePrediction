#1st block
import numpy as np
import pandas as pd
from google.colab import files
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.metrics import accuracy_score
from google.colab import drive
import os

#drive.mount('/content/drive')
ages = pd.read_csv("train.csv")
image_folder = '*your source here*'
file_names = os.listdir('*your source here*')

#***The main task is to predict the age of a person from his or her facial attributes. For simplicity, the problem has been converted to a multiclass problem with classes as Young, Middle and Old.***



#2nd block
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import os

label_mapping = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}
ages['Class'] = ages['Class'].map(label_mapping)

ages_sample = ages.sample(n=5000, random_state=42)

images = []
labels = []

for index, row in ages_sample.iterrows():
    img_path = os.path.join(image_folder, row['ID'])
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array /= 255.0
    images.append(img_array)
    labels.append(row['Class'])

ages_images = np.array(images)
ages_labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(
    ages_images, ages_labels, stratify=ages_labels, test_size=0.25, random_state=42
)

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)



#3rd block
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()




#4th block
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = "relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(3, activation = "softmax"))

model.summary()




#5th block
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(train_images, train_labels, epochs = 30)




#6th block
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]

plt.plot(np.arange(len(train_loss)), train_loss, label = "train loss")
plt.plot(np.arange(len(train_acc)), train_acc, label = "train accuracy")
plt.xlabel("epoch")
plt.legend()




#7th block
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test loss: {test_loss},  test accuracy: {test_acc}")



#8th block
from tensorflow.keras.models import load_model

print(model)

model.save("nn.ages")
model = "*** The model has been deleted. ***"
print(model)

model = load_model("nn.ages")
print(model)

model.summary()




#9th block
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from google.colab.patches import cv2_imshow

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

age_model = load_model("nn.ages")

label_mapping = {0: 'Young', 1: 'Middle', 2: 'Old'}

def predict_age(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    if not img.size:
        print("Error: Loaded image is empty.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        face_roi = cv2.resize(face_roi, (28, 28))
        face_roi = face_roi / 255.0

        face_roi = face_roi.reshape((1, 28, 28, 1))

        predicted_age = np.argmax(age_model.predict(face_roi))

        age_label = label_mapping[predicted_age]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f'Age: {age_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_age('photo.jpg')






#***webcam version***
#*** The code for the webcam version in Google Colab (not functional) ***

# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from google.colab.patches import cv2_imshow

# age_model = load_model("nn.ages")

# # Function to capture a frame from the webcam using JavaScript in Colab
# def capture_frame(filename='photo.jpg', quality=0.8):
#     display(Javascript('''
#         async function captureFrame(quality) {
#             const div = document.createElement('div');
#             const capture = document.createElement('button');
#             capture.textContent = 'Capture Frame';
#             div.appendChild(capture);

#             const video = document.createElement('video');
#             video.style.display = 'block';
#             const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

#             document.body.appendChild(div);
#             div.appendChild(video);
#             video.srcObject = stream;
#             await video.play();

#             capture.onclick = () => {
#                 const canvas = document.createElement('canvas');
#                 canvas.width = video.videoWidth;
#                 canvas.height = video.videoHeight;
#                 const context = canvas.getContext('2d');
#                 context.drawImage(video, 0, 0, canvas.width, canvas.height);
#                 const imgDataUrl = canvas.toDataURL('image/jpeg', quality);
#                 google.colab.kernel.invokeFunction('notebook.run_code', [filename, imgDataUrl]);
#             };

#             return new Promise((resolve) => {
#                 capture.onclick = () => resolve();
#             });
#         }
#         captureFrame();
#     '''))

# def notebook_run_code(filename, imgDataUrl):
#     image_data = b64decode(imgDataUrl.split(',')[1])
#     with open(filename, 'wb') as f:
#         f.write(image_data)
#     print(f'Frame saved as {filename}')

# def predict_age(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Unable to read the image at {image_path}")
#         return

#     if not img.size:
#         print("Error: Loaded image is empty.")
#         return

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y + h, x:x + w]

#         face_roi = cv2.resize(face_roi, (28, 28))
#         face_roi = face_roi / 255.0

#         face_roi = face_roi.reshape((1, 28, 28, 1))

#         predicted_age = np.argmax(age_model.predict(face_roi))

#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(img, f'Age: {predicted_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     cv2_imshow(img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# predict_age('photo.jpg')
# predict_age(capture_frame())
