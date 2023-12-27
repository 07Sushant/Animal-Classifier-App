from flask import Flask, render_template, request, send_from_directory
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
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    try:
        img = request.files['image']
        img.save('static/{}.jpg'.format(COUNT))

        img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
        img_arr = cv2.resize(img_arr, (128, 128))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 128, 128, 3)
        
        prediction = model.predict(img_arr)
        x, y = round(prediction[0, 0], 2), round(prediction[0, 1], 2)
        preds = {'Class 1': x, 'Class 2': y}

        COUNT += 1
        return render_template('prediction.html', data=preds)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('error.html', error=error_message)

@app.route('/load_img')
def load_img():
    global COUNT
    try:
        return send_from_directory('static', "{}.jpg".format(COUNT - 1))
    except FileNotFoundError:
        return render_template('error.html', error="Image not found")

if __name__ == '__main__':
    app.run(debug=True)
