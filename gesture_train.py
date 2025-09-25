import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATASET_DIR = r"C:\Users\nwjoy\OneDrive\Documents\SkillCraft\SCT-4\leapGestRecog"
IMG_SIZE = (64, 64)   # smaller size for speed

X, y, class_names = [], [], []
gesture_folders = set()  # to collect unique gesture folder names

for subject in sorted(os.listdir(DATASET_DIR)):
    subject_path = os.path.join(DATASET_DIR, subject)
    if not os.path.isdir(subject_path):
        continue
    for gesture in sorted(os.listdir(subject_path)):
        gesture_path = os.path.join(subject_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        if gesture not in class_names:
            class_names.append(gesture)
        label_idx = class_names.index(gesture)

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_idx)

X = np.array(X).astype("float32") / 255.0
X = np.expand_dims(X, -1)
y = np.array(y)

print(f"Dataset loaded: {X.shape[0]} samples, {len(class_names)} gesture classes.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_train_cat = to_categorical(y_train, num_classes=len(class_names))
y_test_cat = to_categorical(y_test, num_classes=len(class_names))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=15, batch_size=32
)
model.save("hand_gesture_cnn.h5")

print("Starting live demo... Press ESC to exit.")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    inp = resized.astype("float32") / 255.0
    inp = np.expand_dims(inp, axis=(0, -1))  # shape (1, H, W, 1)

    pred = model.predict(inp, verbose=0)
    class_id = np.argmax(pred)
    prob = np.max(pred)
    label = f"{class_names[class_id]} ({prob*100:.1f}%)"

    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break
cap.release()
cv2.destroyAllWindows()
