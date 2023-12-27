import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# ... Add other layers as in your original code ...

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.load_weights('static/model.h5')

COUNT = 0

def main():
    st.title('Image Classifier')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        global COUNT
        COUNT += 1
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        x, y = round(prediction[0, 0], 2), round(prediction[0, 1], 2)
        preds = {'Class 1': x, 'Class 2': y}

        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Prediction:")
        st.write(preds)

if __name__ == '__main__':
    main()
