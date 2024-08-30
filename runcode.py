import os
import cv2
import pydicom
import threading
import numpy as np
import tkinter as tk
from tqdm import tqdm
import tensorflow as tf
from statistics import mode
from skimage.io import imread
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from joblib import Parallel, delayed
from skimage.transform import resize
import tensorflow.keras.backend as K # type: ignore
from keras.callbacks import LambdaCallback # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models, preprocessing, utils # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation # type: ignore


# Variabile globale
loaded_imgs = []
model = None
best_accuracy = 0.0
history = None
test_img = None
test_mask = None
segmented_img = None
image_files = []
mask_files = []
threshold = 0.90
images = None
masks = None
saved_model_path = "model.h5"
train_dataset = None
train_images = None
train_masks = None
img_tk_test = None
img_tk_segmented = None
canvas_test = None
canvas_segmented = None
batch_size= 8

IMG_HEIGHT = 512
IMG_WIDTH = 512


def build_model(input_shape, learning_rate=0.0001):
    input_images = Input(shape=input_shape)
    input_masks = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(input_images)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    concat1 = Concatenate()([conv1, input_masks])
    pool1 = MaxPooling2D(pool_size=(2, 2))(concat1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Decoder
    up5 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(pool4), conv4])
    conv5 = Conv2D(512, (3, 3), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv3])
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv2])
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv1])
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[input_images, input_masks], outputs=conv9)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy', dice_coefficient])
    return model


def print_dice_coefficient(epoch, logs):
    print(f"Epoch {epoch+1} - dice_coefficient: {logs.get('dice_coefficient')}")

metrics_callback = LambdaCallback(on_epoch_end=print_dice_coefficient)



def preprocess_train_dataset(image_files, mask_files, img_width, img_height):
    processed_images = []
    processed_masks = []

    def process_image(image_file, mask_file):
        try:
            
            image = load_and_preprocess_image(image_file, img_width, img_height)
            mask = load_and_preprocess_mask(mask_file, img_width, img_height)
            if image is not None and mask is not None:
                processed_images.append(image)
                processed_masks.append(mask)
            else:
                print(f"Eroare în procesarea imaginii sau măștii: image_file={image_file}, mask_file={mask_file}")
        except Exception as e:
            print(f"Eroare în procesarea imaginii sau măștii: {e}")

    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        process_image(image_file, mask_file)

    if len(processed_images) == 0 or len(processed_masks) == 0:
        print("Nu s-au procesat perechi valide de imagini și măști.")
        return None, None
    
    return np.array(processed_images), np.array(processed_masks)



def create_data_generator(images, masks, batch_size=8):
    
    if len(images.shape) != 4 or len(masks.shape) != 4:
        raise ValueError("Array-urile de imagini și măști trebuie să aibă patru dimensiuni.")

    image_datagen = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255.
    )

    mask_datagen = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255.
    )

    if image_datagen.rescale != mask_datagen.rescale:
        raise ValueError("Generatoarele de imagini și măști trebuie să aibă același factor de rescalare.")

    def generator():
        image_iterator = image_datagen.flow(images, batch_size=batch_size, seed=42)
        mask_iterator = mask_datagen.flow(masks, batch_size=batch_size, seed=42)
        
        while True:
            image_batch = next(image_iterator)
            mask_batch = next(mask_iterator)
            yield image_batch, mask_batch

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(None, images.shape[1], images.shape[2], images.shape[3]), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, masks.shape[1], masks.shape[2], masks.shape[3]), dtype=tf.float32)
                                             ))
    dataset = dataset.repeat()  
    dataset = dataset.batch(batch_size)

    return dataset



def load_and_preprocess_image(file_path, img_width, img_height):
    print(f"Se încarcă și se preprocesează imaginea: {file_path}")
    try:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Eroare la încărcarea imaginii: {file_path}")
            return None
        
        image_resized = cv2.resize(image, (img_width, img_height))
        print(f"Dimensiunea imaginii redimensionate: {image_resized.shape}")
        

        image_normalized = image_resized / 255.0
        
        image_array = np.expand_dims(image_normalized, axis=-1)
        
        return image_array
    except Exception as e:
        print(f"Eroare la procesarea imaginii: {e}")
        return None



def load_and_preprocess_mask(file_path, img_width, img_height):
    print(f"Se încarcă și se preprocesează masca: {file_path}")
    try:
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Eroare la încărcarea măștii: {file_path}")
            return None
        
        mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        mask_normalized = mask_resized / 255.0
        
        mask_array = np.expand_dims(mask_normalized, axis=-1)
        
        print(f"Dimensiunea măștii redimensionate: {mask_array.shape}")
        return mask_array
    except Exception as e:
        print(f"Eroare la procesarea măștii: {e}")
        return None


def load_image_and_mask():
    global images, masks, train_dataset, train_images, train_masks

    image_folder = filedialog.askdirectory(title="Selectează dosarul cu imagini")
    if not image_folder:
        print("Anulată. Nu a fost selectat niciun dosar cu imagini.")
        return None, None

    mask_folder = filedialog.askdirectory(title="Selectează dosarul cu măști")
    if not mask_folder:
        print("Anulată. Nu a fost selectat niciun dosar cu măști.")
        return None, None

    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')])

    if len(image_files) == 0 or len(mask_files) == 0:
        print("Nu s-au găsit imagini sau măști în dosarele selectate.")
        return None, None

    try:
        train_images, train_masks = preprocess_train_dataset(image_files, mask_files, IMG_WIDTH, IMG_HEIGHT)
        
        if train_images is None or train_masks is None:
            print("Preprocesarea setului de date a eșuat.")
            return None, None

        print("Imagini și măști încărcate și preprocesate.")
        num_images = len(train_images)
        num_masks = len(train_masks)
        print(f"Number of images for training: {num_images}")
        print(f"Number of masks for training: {num_masks}")
        return images, masks
    
    except Exception as e:
        print(f"Eroare în încărcarea și preprocesarea imaginilor și măștilor: {e}")
        return None, None


# Definim funcția Dice Coefficient
def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def train_model(train_images, train_masks, epochs_threshold):
    global model, best_accuracy

    if model is None:
        print("Definiți și compilați modelul înainte de a începe antrenamentul.")
        return None, None

    if train_images is None or train_masks is None:
        print("Eroare: Lipsesc datele de antrenare.")
        return None, None

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', dice_coefficient])

    steps_per_epoch = len(train_images) // batch_size

    def on_epoch_end(epoch, logs):
        current_step = (epoch + 1) * steps_per_epoch
        total_steps = epochs_threshold * steps_per_epoch
        print(f"{current_step}/{total_steps} - loss: {logs['loss']}, accuracy: {logs['accuracy']}")

    progress_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    num_images = len(train_images)
    num_masks = len(train_masks)
    print(f"Numărul de imagini pentru antrenare: {num_images}")
    print(f"Numărul de măști pentru antrenare: {num_masks}")

    history = model.fit(
        x=[train_images, train_masks],
        y=train_masks,
        batch_size=batch_size,
        epochs=epochs_threshold,
        validation_split=0.2, 
        steps_per_epoch=steps_per_epoch,  
        callbacks=[progress_callback]  
    )
   
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    print(f"Acuratețe pe setul de antrenare: {train_accuracy}")
    print(f"Acuratețe pe setul de validare: {val_accuracy}")
   
    if val_accuracy > best_accuracy:
        model.save(saved_model_path)
        print(f"Model salvat cu o acuratețe mai bună pe setul de validare: {val_accuracy}")
        best_accuracy = val_accuracy

    return model, history


def train_button_click():
    global model, train_images, train_masks

    # Construim modelul dacă nu este definit
    if model is None:
        model = build_model((IMG_HEIGHT, IMG_WIDTH, 1))

   
    if train_images is not None and train_masks is not None:
        epochs_threshold = int(epochs_entry.get())
        train_model(train_images, train_masks, epochs_threshold)  
    else:
        
        print("Eroare: Imagini sau măști lipsă.")


def save_model_state():
    global model
    if model:
        model.save('model.h5')
        print("Model salvat cu succes.")
    else:
        print("Modelul nu este definit.")


def load_model_state():
    global model
    model_path = filedialog.askopenfilename(filetypes=[("H5 files", ".h5"), ("Keras files", ".keras")])
    if model_path:
        model = load_model(model_path)
        print(f"Model încărcat cu succes din {model_path}.")
    else:
        print("Încărcarea modelului anulată.")


def calculate_iou(segmented_img, ground_truth_mask):
   
    seg_binary = (segmented_img > 0).astype(np.uint8)
    mask_binary = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(seg_binary, mask_binary)
    union = np.logical_or(seg_binary, mask_binary)

    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def load_test_image():
    global test_img, test_mask

    file_path = filedialog.askopenfilename(title="Selectează imaginea de testare", filetypes=[("Imagini", "*.dcm *.png")])
    if file_path.lower().endswith('.dcm'):
        dicom_data = pydicom.dcmread(file_path)
        test_img = dicom_data.pixel_array
    elif file_path.lower().endswith('.png'):
        test_img = imread(file_path, as_gray=True)
        test_img = (test_img * 255).astype(np.uint8)
    else:
        print("Format de fișier necunoscut.")
        return

    if test_img is not None:
        display_test_image(test_img)

    mask_path = filedialog.askopenfilename(title="Selectează masca de testare (opțional)", filetypes=[("Imagini", "*.png")])
    if mask_path:
        test_mask = imread(mask_path, as_gray=True)
        test_mask = (test_mask * 255).astype(np.uint8)
    else:
        test_mask = None


def segment_test_image(model, test_img, test_mask=None, threshold=0.50):
    try:
        
        resized_img = resize(test_img, (512, 512), anti_aliasing=True)
        normalized_img = (resized_img - np.min(resized_img)) / (np.max(resized_img) - np.min(resized_img))
        x_input = normalized_img.reshape(1, 512, 512, 1)

        if test_mask is not None:
            resized_mask = resize(test_mask, (512, 512), anti_aliasing=True)
            normalized_mask = (resized_mask - np.min(resized_mask)) / (np.max(resized_mask) - np.min(resized_mask))
            y_input = normalized_mask.reshape(1, 512, 512, 1)
        else:
            y_input = np.zeros_like(x_input) 

        
        segmented_img = model.predict([x_input, y_input])[0, :, :, 0]
        print("Dimensiunea imaginii segmentate:", segmented_img.shape)

        print("Valoarea maximă înainte de redimensionare:", np.max(segmented_img))
        print("Valoarea minimă înainte de redimensionare:", np.min(segmented_img))
        print("Primele 10 valori din imaginea segmentată:", segmented_img.flatten()[:10])

        binary_img = (segmented_img >= threshold).astype(np.uint8)

        if test_mask is not None:
            iou_score = calculate_iou(binary_img, test_mask)
            print(f"IoU score: {iou_score}")

        return binary_img

    except Exception as e:
        print(f"Eroare în predicția imaginii: {e}")
        return None



def segment_test_image_thread():
    global model, test_img, test_mask, segmented_img

    if model is None or test_img is None:
        print("Modelul sau imaginea de testare lipsesc.")
        return

    segmented_img = segment_test_image(model, test_img, test_mask)

    if segmented_img is not None:
        print("Segmentarea a fost efectuată cu succes.")

        display_segmented_image(segmented_img)

        apply_threshold()

    else:
        print("Eroare: Nu s-a putut efectua segmentarea imaginii.")


def display_test_image(image):
    global canvas_test, img_tk_test

    if img_tk_test is not None:
        canvas_test.delete(img_tk_test)

    img_resized = Image.fromarray(image).resize((400, 400))

    img_tk_test = ImageTk.PhotoImage(img_resized)

    canvas_test.create_image(0, 0, anchor=tk.NW, image=img_tk_test)
    canvas_test.image = img_tk_test



def display_segmented_image(segmented_img):
    global canvas_segmented, img_tk_segmented

    binary_img = (segmented_img * 255).astype(np.uint8)

    img = Image.fromarray(binary_img, mode='L')
    img_resized = img.resize((400, 400), Image.LANCZOS)

    img_tk_segmented = ImageTk.PhotoImage(img_resized)

    canvas_segmented.create_image(0, 0, anchor=tk.NW, image=img_tk_segmented)
    canvas_segmented.image = img_tk_segmented


def apply_threshold(event=None):
    global segmented_img, threshold_slider, canvas_segmented

    if segmented_img is None:
        print("Nu există o imagine segmentată de aplicat threshold-ul.")
        return

    threshold_value = float(threshold_slider.get())

    binary_segmented_img = np.where(segmented_img > threshold_value, 1, 0)

    display_segmented_image(binary_segmented_img)


def change_threshold(event=None):
    apply_threshold(event)
    
    


root = tk.Tk()
root.title("Cardiac segmentation")

load_model_button = tk.Button(root, text="Load model", command=load_model_state)
load_model_button.grid(row=0, column=0, padx=(2, 2), pady=1, sticky=tk.W)

load_training_button = tk.Button(root, text="Upload training images", command=load_image_and_mask)
load_training_button.grid(row=0, column=1, padx=(2, 2), pady=1, sticky=tk.W)

train_button = tk.Button(root, text="Train model", command=train_button_click)
train_button.grid(row=0, column=2, padx=(2), pady=1, sticky=tk.W)

load_test_button = tk.Button(root, text="Upload test images", command=load_test_image)
load_test_button.grid(row=0, column=3, padx=(2, 2), pady=1, sticky=tk.W)  

segmentation_button = tk.Button(root, text="Segment test image", command=segment_test_image_thread)
segmentation_button.grid(row=0, column=4, padx=(2, 2), pady=1, sticky=tk.W)

epochs_label = tk.Label(root, text="Number of epochs:")
epochs_label.grid(row=1, column=0, padx=3, pady=5, sticky=tk.E)

epochs_entry = tk.Entry(root)
epochs_entry.grid(row=1, column=1, padx=3, pady=5)
epochs_entry.insert(0, "10")

threshold_label = tk.Label(root, text="Threshold:")
threshold_label.grid(row=1, column=2, padx=3, pady=5, sticky=tk.E)

threshold_slider = tk.Scale(root, from_=0, to=1, resolution=0.001, orient=tk.HORIZONTAL, command=change_threshold)
threshold_slider.grid(row=1, column=3, padx=3, pady=5, sticky=tk.W)
threshold_slider.set(0.50)

canvas_test = tk.Canvas(root, width=400, height=400)
canvas_test.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky=tk.W)

canvas_segmented = tk.Canvas(root, width=400, height=400)
canvas_segmented.grid(row=3, column=2, columnspan=2, padx=10, pady=10, sticky=tk.E)

img_tk_orig = None
img_tk_test = None
img_tk_segmented = None

root.mainloop()