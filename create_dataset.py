import os
import nibabel as nib
import numpy as np
import cv2

# input folders
AUTISTIC_PATH = "autistic"
NON_AUTISTIC_PATH = "non-autistic"

# output folders
OUTPUT_PATH = "processed_dataset"
os.makedirs(OUTPUT_PATH + "/autistic", exist_ok=True)
os.makedirs(OUTPUT_PATH + "/non_autistic", exist_ok=True)

IMG_SIZE = 128

def process_folder(input_folder, label_folder):

    count = 0

    for file in os.listdir(input_folder):
        if file.endswith(".nii"):

            filepath = os.path.join(input_folder, file)
            print("Processing:", filepath)

            img = nib.load(filepath)
            data = img.get_fdata()

            total_slices = data.shape[2]

            # take middle 40 slices
            start = total_slices//2 - 20
            end = total_slices//2 + 20

            for i in range(start, end):
                slice_img = data[:, :, i]

                # normalize 0-255
                slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
                slice_img = (slice_img * 255).astype(np.uint8)

                # resize
                slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))

                filename = f"{file}_{i}.png"
                save_path = os.path.join(OUTPUT_PATH, label_folder, filename)

                cv2.imwrite(save_path, slice_img)
                count += 1

    print(label_folder, "images created:", count)


# run for both classes
process_folder(AUTISTIC_PATH, "autistic")
process_folder(NON_AUTISTIC_PATH, "non_autistic")

print("DONE — Dataset Ready")