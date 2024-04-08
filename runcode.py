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

# Variabile globale pentru a stoca imaginile încărcate și imaginea segmentată
loaded_imgs = []  # Lista de imagini încărcate
segmented_img = None
test_img = None  # Imaginea de testare

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

# Funcția pentru încărcarea imaginilor DICOM pentru antrenare
def load_training_images():
    global loaded_imgs
    file_paths = filedialog.askopenfilenames()
    loaded_imgs.clear()  # Golim lista de imagini încărcate anterior
    for file_path in file_paths:
        dicom_data = pydicom.dcmread(file_path)
        loaded_img = dicom_data.pixel_array
        print("Dimensiunea imaginii încărcate:", loaded_img.shape)  # Afișăm dimensiunea imaginii
        loaded_imgs.append(loaded_img)


# Funcția pentru încărcarea imaginii de testare
def load_test_image():
    global test_img
    file_path = filedialog.askopenfilename()
    dicom_data = pydicom.dcmread(file_path)
    test_img = dicom_data.pixel_array

    # Redimensionarea și afișarea imaginii de testare
    if test_img is not None:
        display_test_image(test_img)

# Funcția pentru segmentarea imaginii de testare
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
            
            # Afisarea imaginii segmentate
            pil_img = Image.fromarray((resized_segmented_img * 255).astype(np.uint8))
            display_segmented_image(pil_img)
        else:
            print("Antrenați mai întâi modelul.")
    else:
        print("Încărcați mai întâi o imagine de testare.")

# Funcție pentru antrenarea modelului
def train_model():
    global model, data_generator
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
            y_train_resized.append(resize(loaded_img, (512, 512, 1)))  # Redimensionăm etichetele la aceeași dimensiune cu imaginile

        # Convertim listele în numpy arrays
        x_train_resized = np.array(x_train_resized)
        y_train_resized = np.array(y_train_resized)

        # Antrenarea modelului
        model.fit(data_generator.flow(x_train_resized, y_train_resized, batch_size=1), epochs=epochs)

        print("Antrenarea modelului a fost finalizată.")
    else:
        print("Încărcați mai întâi imagini DICOM pentru antrenare.")

# Funcție pentru antrenarea modelului într-un fir de execuție separat
def train_model_thread():
    threading.Thread(target=train_model).start()

# Funcție pentru afișarea imaginii de antrenare pe Canvas
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
    
    # Convertim matricea numpy în imagine PIL
    img_pil = Image.fromarray(img)

    # Creăm PhotoImage din imaginea PIL
    img_tk_test = ImageTk.PhotoImage(img_pil)
    canvas_test.create_image(0, 0, anchor=tk.NW, image=img_tk_test)

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

# Canvas pentru afișarea imaginii de testare
canvas_test = tk.Canvas(root, width=400, height=400)
canvas_test.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky=tk.W)

# Canvas pentru afișarea imaginii segmentate
canvas_segmented = tk.Canvas(root, width=400, height=400)
canvas_segmented.grid(row=2, column=2, columnspan=2, padx=10, pady=10, sticky=tk.E)

# Inițializăm o referință la imaginea încărcată, la imaginea de testare și la imaginea segmentată
img_tk_orig = None
img_tk_test = None
img_tk_segmented = None

# Rularea buclei principale a aplicației
root.mainloop()