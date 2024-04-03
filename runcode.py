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

# Variabile globale pentru a stoca imaginea încărcată și imaginea segmentată
loaded_img = None
segmented_img = None

# Variabile globale pentru model și generatorul de date
model = None
data_generator = None

# Funcție pentru construirea modelului U-Net
def build_model(input_shape):
    inputs = Input(input_shape)
 
    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Dropout și Batch Normalization pentru regularizare
    conv2 = Dropout(0.5)(conv2)
    conv2 = BatchNormalization()(conv2)
 
    # Decoder
    up3 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv2), conv1])
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
 
    up4 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv3), Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv1)])
    conv4 = Conv2D(1, (1, 1), activation='sigmoid')(up4)
 
    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Funcția pentru încărcarea imaginii DICOM
def load_image():
    global loaded_img
    file_path = filedialog.askopenfilename()
    dicom_data = pydicom.dcmread(file_path)
    loaded_img = dicom_data.pixel_array

    # Redimensionarea imaginii la dimensiunea așteptată de model
    resized_img = resize(loaded_img, (256, 256))

    # Normalizarea imaginii la intervalul [0, 1]
    normalized_img = resized_img / np.max(resized_img)

    # Convertirea imaginii la formatul acceptat de PIL (uint8)
    pil_img = Image.fromarray((normalized_img * 255).astype(np.uint8))

    display_image(pil_img)

# Funcția pentru segmentarea imaginii
def segment_image():
    global loaded_img, model, segmented_img
    if loaded_img is not None:
        if model is not None:
            # Redimensionarea imaginii la dimensiunea așteptată de model
            resized_img = resize(loaded_img, (256, 256))
            normalized_img = resized_img / np.max(resized_img)
            x_input = normalized_img.reshape(1, 256, 256, 1)

            # Segmentarea imaginii folosind modelul
            segmented_img = model.predict(x_input)

            # Redimensionarea imaginii segmentate la dimensiunea imaginii originale
            resized_segmented_img = resize(segmented_img[0, :, :, 0], loaded_img.shape)
            
            # Afisarea imaginii segmentate
            pil_img = Image.fromarray((resized_segmented_img * 255).astype(np.uint8))
            display_segmented_image(pil_img)
        else:
            print("Antrenați mai întâi modelul.")
    else:
        print("Încărcați mai întâi o imagine DICOM.")

# Funcție pentru antrenarea modelului
def train_model():
    global model, data_generator
    epochs = int(epochs_entry.get())  # Obține numărul de epoci din câmpul de intrare
    if loaded_img is not None:
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
        x_train_resized = resize(loaded_img, (256, 256, 1))
        y_train_resized = resize(loaded_img, (512, 512, 1))  # Redimensionăm etichetele la aceeași dimensiune cu imaginile

        # Extindem dimensiunea pe axa 0 pentru a se potrivi cu cerințele generatorului de date
        x_train_resized = np.expand_dims(x_train_resized, axis=0)
        y_train_resized = np.expand_dims(y_train_resized, axis=0)

        # Antrenarea modelului
        model.fit(data_generator.flow(x_train_resized, y_train_resized, batch_size=1), epochs=epochs)

        print("Antrenarea modelului a fost finalizată.")
    else:
        print("Încărcați mai întâi o imagine DICOM.")


# Funcție pentru antrenarea modelului într-un fir de execuție separat
def train_model_thread():
    threading.Thread(target=train_model).start()

# Funcție pentru afișarea imaginii pe Canvas
def display_image(img):
    global canvas_orig, img_tk_orig
    if img_tk_orig is not None:
        canvas_orig.delete(img_tk_orig)
    img_tk_orig = ImageTk.PhotoImage(img)
    canvas_orig.create_image(0, 0, anchor=tk.NW, image=img_tk_orig)

# Funcție pentru afișarea imaginii segmentate pe Canvas
def display_segmented_image(img):
    global canvas_segmented, img_tk_segmented
    if img_tk_segmented is not None:
        canvas_segmented.delete(img_tk_segmented)
    img_tk_segmented = ImageTk.PhotoImage(img)
    canvas_segmented.create_image(0, 0, anchor=tk.NW, image=img_tk_segmented)

# Crearea ferestrei principale
root = tk.Tk()
root.title("Segmentare cardiacă folosind imagini DICOM")

# Butonul pentru încărcarea imaginii DICOM
load_button = tk.Button(root, text="Încarcă imagine DICOM", command=load_image)
load_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

# Butonul pentru antrenarea modelului
train_button = tk.Button(root, text="Antrenare model", command=train_model_thread)
train_button.grid(row=0, column=1, padx=5, pady=5)

# Butonul pentru segmentarea imaginii
segmentation_button = tk.Button(root, text="Segmentare imagine", command=segment_image)
segmentation_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.E)

# Eticheta și câmpul de intrare pentru numărul de epoci
epochs_label = tk.Label(root, text="Număr de epoci:")
epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)

epochs_entry = tk.Entry(root)
epochs_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
epochs_entry.insert(0, "5")  # Setează valoarea implicită a numărului de epoci

# Canvas pentru afișarea imaginii originale
canvas_orig = tk.Canvas(root, width=300, height=300)
canvas_orig.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Canvas pentru afișarea imaginii segmentate
canvas_segmented = tk.Canvas(root, width=300, height=300)
canvas_segmented.grid(row=2, column=2, padx=10, pady=10)

# Inițializăm o referință la imaginea încărcată și la imaginea segmentată
img_tk_orig = None
img_tk_segmented = None

# Rularea buclei principale a aplicației
root.mainloop()
