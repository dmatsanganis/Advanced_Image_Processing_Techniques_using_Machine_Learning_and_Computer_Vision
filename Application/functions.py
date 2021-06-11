from sklearn.cluster import KMeans
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import gabor
from sklearn import preprocessing, svm, metrics
import numpy as np
import cv2


# Function for color space discretization void.
def image_quantization(lab_images, clusters):

    # K-Centroid colors of all images in LAB color space.
    all_colors = []

    # K-Means initialization.
    k_means = KMeans(n_clusters=clusters)

    # Quantized LAB images and RGB images.
    lab_images_q = []
    rgb_images_q = []

    # For every source image converted in LAB color space.
    for lab_image in lab_images:
        # Reshape the source image in order to pass it to K-Means.
        height, width, depth = lab_image.shape
        lab_image_reshaped = lab_image.reshape((height * width, depth))

        # K-means fit the reshaped source image.
        k_means.fit(lab_image_reshaped)

        # Get the k-centroid colors of the images in LAB color space and put them in the list.
        colors_lab = k_means.cluster_centers_.astype('uint8')
        all_colors.extend(colors_lab)

        # Create quantized version of the source image and store it to the list of quantized LAB images.
        labels = k_means.predict(lab_image_reshaped)
        image_q = colors_lab[labels]
        image_q = image_q.reshape((height, width, depth))
        lab_images_q.append(image_q)

        # Convert the quantized LAB images to RGB color space and store it to the list.
        rgb_image_q = cv2.cvtColor(image_q, cv2.COLOR_LAB2BGR)
        rgb_images_q.append(rgb_image_q)

    # Convert the colors list to numpy array.
    all_colors = np.array(all_colors)

    return rgb_images_q, lab_images_q, all_colors, k_means


# Function for SLIC superpixels extraction.
def slic_superpixels(image, image_q, superpixels_number, gray):

    # List for the superpixels of the source image.
    superpixels = []

    # Check if image is grayscale and execute the SLIC algorithm accordingly
    if not gray:
        slic_segments = slic(image_q, n_segments=superpixels_number, start_label=1)
    else:
        slic_segments = slic(image_q, n_segments=superpixels_number,
                        compactness=0.1, sigma=1, start_label=1)

    # For every segment of the SLIC extraction
    for segVal in np.unique(slic_segments):

        # Create black mask to seperate the superpixel and store it to the list
        mask = np.zeros(image_q.shape[:2], dtype="uint8")
        mask[slic_segments == segVal] = 255
        superpixel = cv2.bitwise_and(image_q, image_q, mask=mask)
        superpixels.append(superpixel)

    #-------------------- TESTING ---------------------
    # For displaying the SLIC superpixels of the image
    """
    marked_image = mark_boundaries(image, slic_segments)
    cv2.imshow("SLIC", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #----------------------------------------------------------

    return superpixels


# Function for SURF features extraction.
def surf_features(image, superpixels, gray):

    # List for the descriptors and the keypoints of the source image.
    surf_of_superpixels = []
    keypoints_of_image = []

    # Create SURF object and set it to extended mode for 64 SURF values.
    surf = cv2.xfeatures2d.SURF_create()
    surf.setExtended(True)

    # For each superpixel.
    for superpixel in superpixels:

        # Convert to grayscale if it is not already.
        if not gray:
            gray_superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2GRAY)
        else:
            gray_superpixel = superpixel

        # Store keypoints and descriptors to the lists.
        keypoints, descriptors = surf.detectAndCompute(gray_superpixel, None)
        surf_of_superpixels.append(descriptors)
        keypoints_of_image.extend(keypoints)

        #----------------------- TESTING --------------------------
        # Mark the keypoints on the superpixel and display it.
        """
        surf_superpixel = cv2.drawKeypoints(
            gray_superpixel, keypoints, None, (255, 0, 0))
        cv2.imshow("SURF Feature of superpixel", surf_superpixel)
        cv2.waitKey(0)
        """
        #----------------------------------------------------------

    #----------------------- TESTING --------------------------
    # Mark the keypoints on the whole image and display it
    """
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf_image = cv2.drawKeypoints(image, keypoints_of_image, None, (255, 0, 0))
    cv2.imshow("SURF Feature", surf_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #----------------------------------------------------------

    return surf_of_superpixels


# Function for Gabor feature extraction.
def gabor_features(image, superpixels, kernels, gray):

    # Lists for the Gabor feature of every superpixel and the whole source image.
    gabor_of_superpixels = []
    gabor_of_image = []

    # For every superpixel.
    for superpixel in superpixels:

        # Convert to grayscale if it is not already.
        if not gray:
            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_LAB2BGR)
            gray_superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2GRAY)
        else:
            gray_superpixel = superpixel

        # Apply all gabor kernels to the superpixel and store the result.
        gabor_image, results = apply_kernels(kernels, gray_superpixel)
        gabor_of_superpixels.append(results)

        #----------------------- TESTING --------------------------
        # Display the result of each superpixel.
        """
        cv2.imshow("Gabor Feature of superpixel", gabor_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        #----------------------------------------------------------


    #----------------------- TESTING --------------------------
    # Display the result of the whole image.
    """
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gabor_of_image, _ = apply_kernels(kernels, image)
    cv2.imshow("Gabor Feature", gabor_of_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #----------------------------------------------------------

    return gabor_of_superpixels


# Function for the creation of the kernels.
def create_kernels():

    # List with the created kernels.
    gabor_kernels = []

    # 8 orientations.
    thetas = np.arange(0, np.pi, np.pi / 8)

    # 5 scales.
    lamdas = np.arange(10.0, 60.0, 10.0)

    # Create each kernel and store it to the list.
    ksize = 31
    for theta in thetas:
        for lamda in lamdas:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), 4.0, theta, lamda, 0.5, 0.0, cv2.CV_32F)
            kernel /= 1.5 * kernel.sum()
            gabor_kernels.append(kernel)
    return gabor_kernels


# Function for the application of the kernels to the image.
def apply_kernels(kernels, image):

    # Create a result image with the application of all kernels to the image.
    result_image = np.zeros_like(image)

    # List with the application of each kernel to the image.
    results = []

    # For each kernel.
    for kernel in kernels:

        # Create a filter, apply the kernel to the image and store the result.
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        results.append(filtered)

        # Add to the result image.
        np.maximum(result_image, filtered, result_image)

    return result_image, results


# Function for the creation of the source dataset.
def make_source_dataset(colors_lab, superpixels, surfs, gabors, k_means):

    # Store the A and B values of the quantized colors from the color discretization.
    colors_ab = colors_lab[:, 1:]

    # Create index for every color.
    colors_ab_index = {}
    for index, color in enumerate(colors_ab):
        colors_ab_index[color[0], color[1]] = index

    # Centroid color for each superpixel.
    centroid_colors = []

    # For every image.
    for superpixels_of_image in superpixels:
        # For every superpixel of the image.
        for superpixel in superpixels_of_image:

            # Find all nonzero pixels of the superpixel.
            x_position, y_position, _ = np.nonzero(superpixel)
            pixel = [superpixel[i, j, :]
                     for i, j in zip(x_position, y_position)]
            pixels = np.array(pixel)

            # Find the mean of L, A, B values.
            average_L = np.mean(pixels[:, 0])
            average_a = np.mean(pixels[:, 1])
            average_b = np.mean(pixels[:, 2])

            # Using k-means predict the mean color of the superpixel and store the A, B values.
            label = k_means.predict([[average_L, average_a, average_b]])
            color = colors_lab[label, 1:]
            centroid_colors.append(color)

    # List for the average SURF values.
    surf_average = []

    # For each image.
    for surf_of_image in surfs:
        # For each superpixel of the image.
        for surf in surf_of_image:
            if surf is not None:
                # Find the mean SURF values for each superpixel.
                average = np.mean(surf, axis=0).tolist()
                surf_average.append(average)
            else:
                average = np.zeros(128).tolist()
                surf_average.append(average)

    # List for the average Gabor values.
    gabor_average = []

    # For each image.
    for gabor_of_image in gabors:
        # For each superpixel of the image.
        for gabor_superpixel in gabor_of_image:
            local_average = []
            for gabor in gabor_superpixel:
                # Find the mean Gabor values for each superpixel.
                average = np.mean(gabor[gabor != 0])
                local_average.append(average)
            gabor_average.append(local_average)

    # Find the number of all superpixels of all the images together.
    num_of_superpixels = 0
    for superpixel in superpixels:
        num_of_superpixels = num_of_superpixels + len(superpixel)

    training_set = []
    labels = []

    # For each superpixel.
    for i in range(num_of_superpixels):

        # Get the mean SURF, Gabor values and the centroid color.
        surf_feature = surf_average[i]
        gabor_feature = gabor_average[i]
        color = centroid_colors[i]

        # Store the SURF and Gabor values to the training set and the index of the centroid color to the labels.
        sample = surf_feature + gabor_feature
        training_set.append(sample)
        labels.append(colors_ab_index[color[0, 0], color[0, 1]])

    # Apply regularization to the dataset.
    training_set = preprocessing.scale(training_set)
    return training_set, labels, colors_ab


# Function for the creation of the target dataset.
def make_target_dataset(superpixels, surfs, gabors):

    # List for the average SURF values.
    surf_average = []

    # For every superpixel of the image.
    for surf in surfs:
        if surf is not None:
            # Find the mean SURF values for each superpixel.
            average = np.mean(surf, axis=0).tolist()
            surf_average.append(average)
        else:
            average = np.zeros(128).tolist()
            surf_average.append(average)

    # List for the average Gabor values.
    gabor_average = []

    # For every superpixel of the image.
    for gabor_superpixel in gabors:
        local_average = []
        for gabor in gabor_superpixel:
            # Find the mean Gabor values for each superpixel.
            average = np.mean(gabor[gabor != 0])
            local_average.append(average)
        gabor_average.append(local_average)

    testing_set = []

    # For each superpixel.
    for i in range(len(superpixels)):
        # Get the mean SURF and Gabor values.
        surf_feature = surf_average[i]
        gabor_feature = gabor_average[i]

        # Store the SURF and Gabor values to the testing set.
        sample = surf_feature + gabor_feature
        testing_set.append(sample)

    # Apply regularization to the dataset.
    testing_set = preprocessing.scale(testing_set)
    return testing_set


# Function for the training of the SVM.
def svm_model_training(training_set, labels):

    # Create a new SVM and train it on training set and labels.
    s_v_m = svm.SVC()
    s_v_m.fit(training_set, labels)

    # Predict and compute accuracy
    predictions = s_v_m.predict(training_set)
    print("Accuracy Score: " + str(metrics.accuracy_score(labels, predictions)))
    return s_v_m

# Colorize target function.
def colorize_target(svm, testing_set, colors_ab, target_image, target_superpixels):

    # Predict the index of the color using the SVM for the testing set.
    labels = svm.predict(testing_set)

    # Get the A, B values of the color with the predicted index.
    color_labels = colors_ab[labels]

    # Create a blank image with the size of the target image.
    colored_image = np.zeros((target_image.shape[0], target_image.shape[1], 3), dtype='uint8')

    # For every superpixel of the image.
    for index, superpixel in enumerate(target_superpixels):
        # Find all nonzero pixels of the superpixel.
        x_position, y_position = np.nonzero(superpixel)

        # For every nonzero pixel of the superpixel.
        for i, j in zip(x_position, y_position):

            # Colorize according to the predicted values.
            L = target_image[i, j]
            a = color_labels[index, 0]
            b = color_labels[index, 1]

            colored_image[i, j, 0] = L
            colored_image[i, j, 1] = a
            colored_image[i, j, 2] = b

    # Convert the colorized image from LAB to RGB color space.
    colored_image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_LAB2BGR)
    return colored_image, colored_image_rgb