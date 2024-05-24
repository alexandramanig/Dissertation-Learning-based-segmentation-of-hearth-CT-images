import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.transform import resize
import threading
import pydicom
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV
import json
import h5py


# Global variables to store loaded images, segmented image, and model
loaded_imgs = []  
segmented_img = None
test_img = None  
model = None
data_generator = None
best_accuracy = 0.0
history = None  # Define history globally

# Variabile globale pentru model și generatorul de date
model = None
data_generator = None

# Variabila globala pentru stocarea acuratetii modelului
best_accuracy = 0.0

# Functie pentru construirea modelului U-Net
def build_model(input_shape, learning_rate=0.0001):
    # Definirea modelului
    inputs = Input(input_shape)
   # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # Definim pool1 aici

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)
    pool3 = BatchNormalization()(pool3)

    #   Decoder
    up4 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(pool3), conv3])
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(up4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(conv4)
    conv4 = Dropout(0.25)(conv4)
    conv4 = BatchNormalization()(conv4)

    up5 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv4), conv2])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(up5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(conv5)
    conv5 = Dropout(0.25)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv1])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer='l2')(conv6)
    conv6 = Dropout(0.25)(conv6)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2D(1, (1, 1), activation='sigmoid')(conv6)

    model = Model(inputs=[inputs], outputs=[conv7])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Functie pentru încărcarea imaginilor DICOM pentru antrenare
def load_training_images():
    global loaded_imgs
    file_paths = filedialog.askopenfilenames()
    loaded_imgs.clear()  # Golim lista de imagini încărcate anterior
    for file_path in file_paths:
        dicom_data = pydicom.dcmread(file_path)
        loaded_img = dicom_data.pixel_array
        print("Dimensiunea imaginii încărcate:", loaded_img.shape)  # Afișăm dimensiunea imaginii
        loaded_imgs.append(loaded_img)

# Functie pentru încărcarea imaginii de testare
def load_test_image():
    global test_img
    file_path = filedialog.askopenfilename()
    dicom_data = pydicom.dcmread(file_path)
    test_img = dicom_data.pixel_array

    # Redimensionarea și afișarea imaginii de testare
    if test_img is not None:
        display_test_image(test_img)

# Funcție pentru segmentarea imaginii de testare
def segment_test_image():
    global test_img, model, segmented_img
    if test_img is not None:
        if model is not None:
            # Redimensionarea imaginii de testare la dimensiunea așteptată de model
            resized_img = resize(test_img, (256, 256))
            normalized_img = resized_img / np.max(resized_img)
            x_input = normalized_img.reshape(1, 256, 256, 1)

            # Segmentarea imaginii folosind modelul
            segmented_img = model.predict(x_input)

            # Redimensionarea imaginii segmentate la dimensiunea imaginii originale
            resized_segmented_img = resize(segmented_img[0, :, :, 0], test_img.shape)

            # Normalizarea și aplicarea pragului pe imaginea segmentată
            normalized_segmented_img = normalize_segmented_image(resized_segmented_img)
            apply_threshold(normalized_segmented_img)  # Actualizăm apelul pentru a furniza imaginea segmentată
        else:
            print("Antrenați mai întâi modelul.")
    else:
        print("Încărcați mai întâi o imagine de testare.")


# Funcție pentru antrenarea modelului
def train_model():
    global model, data_generator, best_accuracy, history
    global threshold_slider  # adaugăm threshold_slider ca o variabilă globală

    epochs = int(epochs_entry.get())  # Obține numărul de epoci din câmpul de intrare
    if loaded_imgs:
        input_shape = (256, 256, 1)

        # Construirea modelului
        model = build_model(input_shape)

        # Configurarea generatorului de date pentru imagini și etichete
        data_generator = ImageDataGenerator(rotation_range=180,
                                            zoom_range=0.0,
                                            width_shift_range=0.0,
                                            height_shift_range=0.0,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        
        # Redimensionăm imaginile la dimensiunile așteptate de model
        x_train_resized = []
        y_train_resized = []
        for loaded_img in loaded_imgs:
            x_train_resized.append(resize(loaded_img, (256, 256, 1)))
            y_train_resized.append(resize(loaded_img, (256, 256, 1)))  # Redimensionăm etichetele la aceeași dimensiune cu imaginile

        # Convertim listele în numpy arrays
        x_train_resized = np.array(x_train_resized)
        y_train_resized = np.array(y_train_resized)

        # Definim o funcție pentru afișarea progresului în timpul antrenamentului
        def on_epoch_end(epoch, logs):
            print(f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']}, accuracy: {logs['accuracy']}")

        # Adăugăm un callback pentru afișarea progresului
        progress_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        # Antrenarea modelului
        history = model.fit(data_generator.flow(x_train_resized, y_train_resized, batch_size=1), 
                            epochs=epochs, 
                            callbacks=[progress_callback])  # Adăugăm callback-ul pentru afișarea progresului

        # Salvarea modelului doar dacă acuratețea este mai bună
        current_accuracy = history.history['accuracy'][-1]
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            save_model_state()

        print("Antrenarea modelului a fost finalizată.")
    else:
        print("Încărcați mai întâi imagini DICOM pentru antrenare.")

# Functie pentru antrenarea modelului într-un fir de execuție separat
def train_model_thread():
    threading.Thread(target=train_model).start()

# Functie pentru afișarea imaginii de antrenare pe Canvas
def display_image(img):
    global canvas_orig, img_tk_orig
    if img_tk_orig is not None:
        canvas_orig.delete(img_tk_orig)
    
    # Convertim matricea numpy în imagine PIL
    img_pil = Image.fromarray(img)

    # Creăm PhotoImage din imaginea PIL
    img_tk_orig = ImageTk.PhotoImage(img_pil)
    canvas_orig.create_image(0, 0, anchor=tk.NW, image=img_tk_orig)

# Funcție pentru afișarea imaginii de testare pe Canvas
def display_test_image(img):
    global canvas_test, img_tk_test
    if img_tk_test is not None:
        canvas_test.delete(img_tk_test)
    
    # Redimensionare imagine la 256x256
    img_resized = Image.fromarray(img).resize((256, 256))
    
    # Creăm PhotoImage din imaginea redimensionată
    img_tk_test = ImageTk.PhotoImage(img_resized)
    canvas_test.create_image(0, 0, anchor=tk.NW, image=img_tk_test)

# Funcție pentru afișarea imaginii segmentate pe Canvas
def display_segmented_image(img):
    global canvas_segmented, img_tk_segmented

    # Redimensionare imagine la 256x256
    img_resized = Image.fromarray(img).resize((256, 256))
    
    # Converitm imaginea PIL într-un tablou NumPy
    img = np.array(img_resized)

    # Converitm la scala de gri, dacă este necesar
    if len(img.shape) == 3:
        img = img[:, :, 0]

    # Verificăm tipul de date
    if not (img.dtype == np.float32 or img.dtype == np.uint8):
        raise TypeError("Tipul de date al imaginii nu este corect.")

    # Aplicăm pragul bazat pe valoarea glisorului
    threshold = threshold_slider.get()
    binary_img = (img > threshold).astype(np.uint8)

    # Inversăm imaginile binare
    inverted_img = 1 - binary_img

    # Creăm imaginea PIL și o afișăm
    img_pil = Image.fromarray((inverted_img * 255).astype(np.uint8))
    img_tk_segmented = ImageTk.PhotoImage(img_pil)
    canvas_segmented.create_image(0, 0, anchor=tk.NW, image=img_tk_segmented)


# Functie pentru normalizarea imaginii segmentate
def normalize_segmented_image(img):
    img_normalized = img / np.max(img)
    return img_normalized

# Funcție pentru salvarea stării modelului și a datelor istorice
def save_model_state():
    global model, best_accuracy, history
    model.save('model.h5')
    with open('best_accuracy.json', 'w') as f:
        json.dump(best_accuracy, f)
    with open('history.json', 'w') as f:
        json.dump(history.history, f)

# Functie pentru incarcarea starii modelului
def load_model_state():
    global model, best_accuracy, history
    model_weights_file = filedialog.askopenfilename(title="Selectați fișierul cu modelul")
    if model_weights_file:
        model = load_model(model_weights_file)
        with open('best_accuracy.json', 'r') as f:
            best_accuracy = json.load(f)
        with open('history.json', 'r') as f:
            history = json.load(f)
    else:
        print("Nu a fost selectat niciun fișier.")
        
              
# Funcție pentru aplicarea threshold-ului pe imaginea segmentată
def apply_threshold(segmented_img):
    global threshold_slider, canvas_segmented

    if segmented_img is not None:
        # Obținem valoarea threshold-ului din label
        threshold_value = float(threshold_slider.get())

        # Aplicăm threshold-ul pe imaginea segmentată
        thresholded_segmented_img = (segmented_img > threshold_value).astype(np.uint8)

        # Afisăm imaginea segmentată binară
        display_segmented_image(thresholded_segmented_img)
    else:
        print("Nu există o imagine segmentată de aplicat threshold-ul.")


def change_threshold(val):
    threshold_value = float(val)  # convertim valoarea din string în float
    apply_threshold()  # aplicăm threshold-ul pe imaginea segmentată


# Functie pentru optimizarea parametrilor
def optimize_parameters(x_train_resized, y_train_resized):
    param_grid = {'batch_size': [8, 16, 32], 'learning_rate': [0.0001, 0.001, 0.01]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(x_train_resized, y_train_resized)
    print("Parametrii optimi:", grid_search.best_params_)


# Crearea ferestrei principale
root = tk.Tk()
root.title("Segmentare cardiacă folosind imagini DICOM")

# Butonul pentru încărcarea imaginilor DICOM pentru antrenare
load_training_button = tk.Button(root, text="Încarcă imagini antrenare", command=load_training_images)
load_training_button.grid(row=0, column=0, padx=(5, 2), pady=1, sticky=tk.W)  

# Butonul pentru antrenarea modelului
train_button = tk.Button(root, text="Antrenare model", command=train_model_thread)
train_button.grid(row=0, column=1, padx=2, pady=1) 

# Butonul pentru încărcarea imaginii de testare
load_test_button = tk.Button(root, text="Încarcă imagine de testare", command=load_test_image)
load_test_button.grid(row=0, column=2, padx=2, pady=1)  

# Butonul pentru segmentarea imaginii de testare
segmentation_button = tk.Button(root, text="Segmentare imagine de testare", command=segment_test_image)
segmentation_button.grid(row=0, column=3, padx=(2, 5), pady=1, sticky=tk.E)  

# Eticheta și câmpul de intrare pentru numărul de epoci
epochs_label = tk.Label(root, text="Număr de epoci:")
epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)

epochs_entry = tk.Entry(root)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)
epochs_entry.insert(0, "5")  # Setează valoarea implicită a numărului de epoci

# Label și slider pentru threshold
threshold_label = tk.Label(root, text="Threshold:")
threshold_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)

# Setarea valorii implicite pentru threshold_slider
threshold_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=change_threshold)
threshold_slider.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
threshold_slider.set(0.5)  # Setăm valoarea implicită la 0.5

# Buton pentru încărcarea stării modelului salvate anterior
load_model_button = tk.Button(root, text="Încarcă model", command=load_model_state)
load_model_button.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

# Buton pentru optimizarea parametrilor
optimize_button = tk.Button(root, text="Optimizează parametrii", command=lambda: optimize_parameters(x_train_resized, y_train_resized))
optimize_button.grid(row=2, column=1, padx=5, pady=5, sticky=tk.E)

# Canvas pentru afișarea imaginii de testare
canvas_test = tk.Canvas(root, width=400, height=400)
canvas_test.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky=tk.W)

# Canvas pentru afișarea imaginii segmentate
canvas_segmented = tk.Canvas(root, width=400, height=400)
canvas_segmented.grid(row=3, column=2, columnspan=2, padx=10, pady=10, sticky=tk.E)

# Inițializăm o referință la imaginea încărcată, la imaginea de testare și la imaginea segmentată
img_tk_orig = None
img_tk_test = None
img_tk_segmented = None

# Rularea buclei principale a aplicației
root.mainloop()
