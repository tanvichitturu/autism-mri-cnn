import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf

IMG_SIZE = 128

# load trained model
model = tf.keras.models.load_model("autism_mri_model.h5")

def predict_mri(mri_path):

    img = nib.load(mri_path)
    data = img.get_fdata()

    total_slices = data.shape[2]
    start = total_slices//2 - 20
    end = total_slices//2 + 20

    predictions = []

    for i in range(start, end):
        slice_img = data[:, :, i]

        # normalize
        slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
        slice_img = (slice_img * 255).astype(np.uint8)

        # resize
        slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))

        # reshape for CNN
        slice_img = slice_img / 255.0
        slice_img = np.reshape(slice_img, (1, IMG_SIZE, IMG_SIZE, 1))

        pred = model.predict(slice_img, verbose=0)[0][0]
        predictions.append(pred)

    final_score = np.mean(predictions)

    print("Average prediction score:", final_score)

    if final_score > 0.5:
        print("Prediction: AUTISTIC")
    else:
        print("Prediction: NON-AUTISTIC")


# test on a file
predict_mri("sample.nii")