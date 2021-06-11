import cv2
import functions
import numpy as np

NUM_OF_DATASET_IMAGES = 4
SUPERPIXELS = 40
CLUSTERS = 4
TARGET_IMAGE = "Dataset\\target.jpg"

rgb_images = []
lab_images = []
slic_list = []
surf_list = []
gabor_kernels = []
gabor_list = []

# Get the RGB images from folder Dataset
# and store them to list rgb_images.
print("\nProcess: Collection of the RGB source images.")
for i in range(1, (NUM_OF_DATASET_IMAGES + 1)):
    rgb_image = cv2.imread("Dataset\\" + str(i) + ".jpg")
    rgb_images.append(rgb_image)
print("\nCompleted: Collection of the RGB source images.")

# Change the color space of the images from RGB to LAB
# and then store the converted images to list lab_images.
print("\nProcess: Conversion of images to LAB color space.")
for rgb_image in rgb_images:
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
    lab_images.append(lab_image)
print("\nCompleted: Conversion of images to LAB color space.")

# Color space discretization process.
print("\nProcess: Color space discretization.")
rgb_images_q, lab_images_q, lab_colors, k_means = functions.image_quantization(
    lab_images, CLUSTERS)
print("\nCompleted: Color space discretization.")

# Creates 40 GABOR filters with 8 orientations and 5 scales.
gabor_kernels = functions.create_kernels()

# For every source image in the dataset.
for i in range(len(rgb_images)):

    # SLIC superpixels extraction.
    print("\nProcess: Extract SLIC Superpixels for image " + str(i+1) + ".")
    slic = functions.slic_superpixels(rgb_images_q[i], rgb_images[i], SUPERPIXELS, False)
    slic_list.append(slic)
    print("\nCompleted: Extract SLIC Superpixels for image " + str(i+1) + ".")

    # SURF feature extraction.
    print("\nProcess: Extract SURF feature for image " + str(i+1) + ".")
    surf = functions.surf_features(rgb_images[i], slic_list[i], False)
    surf_list.append(surf)
    print("\nCompleted: Extract SURF feature for image " + str(i+1) + ".")

    # Gabor feature extraction.
    print("\nProcess: Extract Gabor feature for image " + str(i+1) + ".")
    gabor = functions.gabor_features(rgb_images[i], slic_list[i], gabor_kernels, False)
    gabor_list.append(gabor)
    print("\nCompleted: Extract Gabor feature for image " + str(i+1) + ".")


# Create dataset for source images, in order to train the model.
print("\nProcess: Creation of the source Dataset.")
training_set, labels, colors_ab = functions.make_source_dataset(lab_colors, slic_list, surf_list, gabor_list, k_means)
print("\nCompleted: Creation of the source Dataset.")

# SVM model training.
print("\nProcess: SVM model training.")
svm = functions.svm_model_training(training_set, labels)
print("\nCompleted: SVM model training.")

# Get target image via open-cv library.
print("\nProcess: Collection of the RGB target image.")
target_image = cv2.imread(TARGET_IMAGE)
print("\nCompleted: Collection of the RGB target image.")

#Convert target image to grayscale.
print("\nProcess : Conversion of the target image to grayscale.")
target_image_g = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
print("\nCompleted : Conversion of the target image to grayscale.")

# SLIC Superpixels extraction process for target image.
print("\nProcess: Extract SLIC Superpixels for the target image.")
target_slic = functions.slic_superpixels(target_image_g, target_image_g, SUPERPIXELS, True)
print("\nCompleted: Extract SLIC Superpixels for the target image.")

# SURF feature extraction process for target image.
print("\nProcess: Extract SURF feature for the target image.")
target_surf = functions.surf_features(target_image_g, target_slic, True)
print("\nCompleted: Extract SURF feature for the target image.")

# Gabor feature extraction process for target image.
print("\nProcess: Extract Gabor feature for target image.")
target_gabor = functions.gabor_features(target_image_g, target_slic, gabor_kernels, True)
print("\nCompleted: Extract Gabor feature for target image.")

# Create dataset process for target image.
print("\nProcess: Creation of the target dataset.")
testing_set = functions.make_target_dataset(target_slic, target_surf, target_gabor)
print("\nCompleted: Creation of the target dataset.")

# Colorize target image process.
print("\nProcess: Colorization of the target image.")
colored_image_lab, colored_image_rgb = functions.colorize_target(svm, testing_set, colors_ab, target_image_g, target_slic)
print("\nCompleted: Colorization of the target image.")

# Display Results to user via pop up window.
target_image_g = cv2.cvtColor(target_image_g, cv2.COLOR_GRAY2BGR)
up = np.hstack((target_image, target_image_g))
down = np.hstack((colored_image_lab, colored_image_rgb))
window = np.vstack((up, down))
# Show the results via a 4-splitted window.
cv2.imshow("Original RGB - Original Grayscale - Colorized LAB - Colorized RGB", window)
cv2.waitKey(0)
cv2.destroyAllWindows()