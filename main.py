import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.ndimage as ndi

# Regarding point 1 and it's requirements =>
# Read the dataset of brain folder using imageio.volread() function

# make a loop to make user enter the required task number

# 1.Provided dicom images, be able to :
# 1.1. Get patient information and Image information.
# 1.2.Get Shape, Sampling, Pixel Aspect ratio and field of viewof the images used.
# 1.3.Slice the image to view from different planes(Explore medical image)

# Carry out image Analysis Operations :
# 2.1.Enhance your medical image by applying appropriate morphological operations (draw the histogram before and after the operation, write your conclusion -at least 4 operations).
# 2.2. Apply your own filter & mask for feature detection.
# 2.3. Extract and segment object of interest (ex. Tumor) from your medical image.
# 2.4. Measure how big the object of interest is, measure distance transformation, center mass of the object of interest.

def task_one():
    # 1.Provided dicom images, be able to :
    # 1.1. Get patient information and Image information.
    # 1.2.Get Shape, Sampling, Pixel Aspect ratio and field of viewof the images used.
    # 1.3.Slice the image to view from different planes(Explore medical image)
    # Read the dataset of brain folder using imageio.volread() function
    vol = imageio.volread("SE000001", format='dicom') #3D
    # Get patient information
    print("Patient Name: ", vol.meta['PatientName'])
    print("Patient ID: ", vol.meta['PatientID'])
    print("Patient Birthdate: ", vol.meta['PatientBirthDate'])
    print("Patient Sex: ", vol.meta['PatientSex'])
    print("Patient Weight: ", vol.meta['PatientWeight'])

    # Get image information
    print("Image Position: ", vol.meta['ImagePositionPatient'])
    print("Image Orientation: ", vol.meta['ImageOrientationPatient'])

    # Print the  number of slices of each plane.
    n0, n1, n2 = vol.shape
    print("the shape is: ", vol.shape, "\n")

    print("Number of Slices:\n\t", "Axial=", n0, "Slices\n\t",
          "Coronal=", n1, "Slices\n\t",
          "Sagittal=", n2, "Slices")

    # Print the sampling rate of each plane, you can find it in the metadata using 'sampling' as a keyword.
    d0, d1, d2 = vol.meta['sampling']

    print("Sampling Rate Axial", d0, "\nSampling Rate Coronal", d1, "\nSampling Rate Sagittal", d2)

    # Print the field of view of each plane by multiplying the number of slides and the sampling rate.
    print("FoV Axial", d0 * n0, "\nFoV Coronal", d1 * n1, "\nFoV Sagittal", d2 * n2)

    # Calculate the ascpect ratio of each plane and save each in a separate variable.

    axial_asp = d1 / d2
    sagittal_asp = d0 / d1
    coronal_asp = d0 / d2

    print("Axial Aspect Ratio", axial_asp, "\nCoronal Aspect Ratio", coronal_asp, "\nSagittal Aspect Ratio",
          sagittal_asp)

    # Define the function that shows the images of the specified slice number.
    def slicer(axial_slice, coronal_slice, sagittal_slice):
        # call the subplots function to create a subplot of 1 row and 3 cols

        fig, ax = plt.subplots(1, 3, figsize=(12, 12))

        # Show the specfied slice on the axial plane "first index" with 'gray' color-map
        # make sure to turn the axis off and specify an appropriate title

        ax[0].imshow(vol[axial_slice, :, :], cmap='gray', aspect=axial_asp)
        ax[0].axis('off')
        ax[0].set_title('axial frame')
        # Show the specfied slice on the coronal plane "second index" with 'gray' color-map
        # make sure to turn the axis off and specify an appropriate title

        ax[1].imshow(vol[:, coronal_slice, :], cmap='gray', aspect=coronal_asp)
        ax[1].axis('off')
        ax[1].set_title('coronal frame')

        # Show the specfied slice on the sagittal plane "third index" with 'gray' color-map
        # make sure to turn the axis off and specify an appropriate title

        ax[2].imshow(vol[:, :, sagittal_slice], cmap='gray', aspect=sagittal_asp)
        ax[2].axis('off')
        ax[2].set_title('sagittal frame')

        # Render the images
        plt.show()

    slicer(5, 40, 100)


def EnhanceImage(im):
    print('Data type:', im.dtype)  # attribute
    print('Min. value:', im.min())
    print('Max value:', im.max())
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    # Show the color bar and render the plot
    plt.colorbar()
    plt.show()

    # Import SciPy's "ndimage" module
    import scipy.ndimage as ndi

    # Create a histogram, binned at each possible value
    hist = ndi.histogram(im, min=0, max=255, bins=256)
    plt.plot(hist)
    plt.suptitle("Original Histogram")
    plt.show()
    # mask to be true for values more than 70 (for bones only)
    mask2 = im > 70  # Bone mask

    # Apply 4 morphological operations which are (dilation, erosion, opening and closing).
    # Tuning : Apply the ndi.binary_dilation function with 10 iterations on mask 2 (bones).
    mask_dilate = ndi.binary_dilation(mask2, iterations=10)
    dilated_im = np.where(mask_dilate, im, 0)

    # Plot The original mask, the dilated mask and the image with the applied dilated mask (using np.where) and the histogram
    # of the dilated image

    fig, axes = plt.subplots(1, 3, figsize=(12, 12))

    axes[0].imshow(mask2 * 255, cmap='gray')
    axes[1].imshow(mask_dilate * 255, cmap='gray')
    axes[2].imshow(dilated_im, cmap='gray')

    for i in range(3):
        axes[i].axis('off')
    plt.show()

    hist_dilate = ndi.histogram(dilated_im, min=0, max=255, bins=256)
    plt.plot(hist_dilate)
    plt.suptitle("Histogram of dilated image")
    plt.show()

    # Tuning : Apply the ndi.binary_erosion function with 10 iterations on mask 2 (bones).
    mask_erosion = ndi.binary_erosion(mask2, iterations=10)
    eroded_im = np.where(mask_erosion, im, 0)

    # Plot The original mask, the eroded mask and the image with the applied dilated mask (using np.where)
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))

    axes[0].imshow(mask2 * 255, cmap='gray')
    axes[1].imshow(mask_erosion * 255, cmap='gray')
    axes[2].imshow(eroded_im, cmap='gray')

    for i in range(3):
        axes[i].axis('off')
    plt.show()

    hist_erosion = ndi.histogram(eroded_im, min=0, max=255, bins=256)
    plt.plot(hist_erosion)
    plt.suptitle("Histogram of eroded image")
    plt.show()

    # Tuning : Apply the ndi.binary_opening function with 10 iterations on mask 2 (bones).
    mask_open = ndi.binary_opening(mask2, iterations=10)
    opened_im = np.where(mask_open, im, 0)

    # Plot The original mask, the opened mask and the image with the applied dilated mask (using np.where)
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))

    axes[0].imshow(mask2 * 255, cmap='gray')
    axes[1].imshow(mask_open * 255, cmap='gray')
    axes[2].imshow(opened_im, cmap='gray')

    for i in range(3):
        axes[i].axis('off')
    plt.show()

    hist_open = ndi.histogram(opened_im, min=0, max=255, bins=255)
    plt.plot(hist_open)
    plt.suptitle("Histogram of opening image")
    plt.show()

    # Tuning : Apply the ndi.binary_closing function with 10 iterations on mask 2 (bones).
    mask_close = ndi.binary_closing(mask2, iterations=10)
    closed_im = np.where(mask_close, im, 0)

    # Plot The original mask, the closed mask and the image with the applied dilated mask (using np.where)
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))

    axes[0].imshow(mask2 * 255, cmap='gray')
    axes[1].imshow(mask_close * 255, cmap='gray')
    axes[2].imshow(closed_im, cmap='gray')

    for i in range(3):
        axes[i].axis('off')
    plt.show()

    hist_close = ndi.histogram(closed_im, min=0, max=255, bins=255)
    plt.plot(hist_close)
    plt.suptitle("Histogram of closing image")
    plt.show()


def ApplyFilter(im):
    # 2) B
    # Set sharping filter weights
    weights = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    # Convolve the image with the filter
    im_filt = ndi.convolve(im, weights)

    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes[0].imshow(im, cmap='gray')
    axes[1].imshow(im_filt, cmap='gray', vmin=0, vmax=255)
    for i in range(2):
        axes[i].axis('off')
    plt.show()

    im_s1 = ndi.gaussian_filter(im, sigma=1)
    im_s3 = ndi.gaussian_filter(im, sigma=3)

    # Draw bone masks of each image
    fig, axes = plt.subplots(3, 1)
    axes[0].imshow(im, cmap='gray')
    axes[1].imshow(im_s1, cmap='gray')
    axes[2].imshow(im_s3, cmap='gray')
    for i in range(3):
        axes[i].axis('off')
    plt.show()

    # Set weights to detect horizontal edges
    weights1 = [[+1, +1, +1],
                [0, 0, 0],
                [-1, -1, -1]]

    # Convolve "im" with filter weights
    edges = ndi.convolve(im, weights1)

    # Draw the image in color
    plt.imshow(edges, cmap='gray', vmin=0, vmax=150)
    plt.colorbar()
    plt.show()

    # Apply Sobel filter along both axes
    sobel_ax0 = ndi.sobel(im, axis=0)
    sobel_ax1 = ndi.sobel(im, axis=1)

    # Calculate edge magnitude
    edges = np.sqrt(np.square(sobel_ax0) + np.square(sobel_ax1))
    # Plot edge magnitude
    plt.imshow(edges, cmap='gray', vmax=255)
    plt.axis('off')
    plt.show()


def ExtractSegment(im):
    # Smooth intensity values
    im_filt = ndi.median_filter(im, size=3)
    # Masking High intensity values

    mask_start = np.where(im_filt > 60, 1, 0)
    mask = ndi.binary_closing(mask_start)

    # Label the objects

    labels, nlabels = ndi.label(mask)
    print("Num of labels", nlabels)

    overlay = np.where(labels > 0, labels, np.nan)
    plt.imshow(overlay, cmap='rainbow')
    plt.axis('off')
    plt.show()

    # Masking the object of interest
    lv_mask = np.where(labels == 81, im, np.nan)
    plt.imshow(lv_mask)
    plt.axis('off')
    plt.show()

    # Finding Bounding Box of LV

    lv_box = ndi.find_objects(labels == 81)
    print("Length of the box: ", len(lv_box))
    print("The box: ", lv_box[0])

    im_lv = im[lv_box[0]]
    plt.imshow(im_lv)
    plt.axis('off')
    plt.show()

    # Measuring Intensities

    print("mean of the image: ", ndi.mean(im))
    print("mean of the image and the labels: ", ndi.mean(im, labels))
    print("mean of the image and specific label: ", ndi.mean(im, labels, index=81))
    print("mean of the image and specific labels: ", ndi.mean(im, labels, index=[79, 81]))


def MeasureObject(im):
    # 2) D
    # area of LV object of interest
    # Smooth intensity values
    im_filt = ndi.median_filter(im, size=3)
    # Masking High intensity values

    mask_start = np.where(im_filt > 60, 1, 0)
    mask = ndi.binary_closing(mask_start)
    labels, nlabels = ndi.label(mask)
    d1, d2 = im.meta['sampling']
    dpix = d1 * d2

    npixels = ndi.sum(1, labels, index=81)
    area = npixels * dpix

    print("Area of LV is ", area)

    # calculate distance transformation

    iv = np.where(labels == 81, 1, 0)
    dists = ndi.distance_transform_edt(iv, sampling=im.meta['sampling'])
    print("Max distance is ", ndi.maximum(dists))
    print("Max location is ", ndi.maximum_position(dists))

    # center of mass
    com = ndi.center_of_mass(im, labels, index=81)
    print("Center of mass is ", com)


while True:
    print(" enter the number of the task you want to do : ")
    print(" if you want to stop the program enter S ")
    task = input()
    if task == "1":
        task_one()

    elif task == "2":
        print("enter the number of the operation you want to do : ")
        print("a. Enhance your medical image by applying appropriatemorphological operations (draw the histogram before and after the operation, write your conclusion -at least 4 operations).")
        print("b. Apply your own filter & mask for feature detection.")
        print("c. Extract and segment object of interest (ex. Tumor) from your medical image. ")
        print("d. Measure how big the object of interest is, measure distance transformation, center mass of the object of interest.")

        operation = input()
        im = imageio.imread("chest-220.dcm")
        if operation == "a":
            EnhanceImage(im)
        elif operation == "b":
            ApplyFilter(im)
        elif operation == "c":
            ExtractSegment(im)
        elif operation == "d":
            MeasureObject(im)

    elif task == 's':
        break
    else:
        print("Invalid input. Please try again.")

