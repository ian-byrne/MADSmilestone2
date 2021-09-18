import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from PIL import Image


def get_images_test(id_rounds):
    counter = 0
    store_images = []
    image_data = []

    for id, array in id_rounds.items():
        for value in array:
            if counter < 10:
                url = (
                    "https://clockimages.s3.us-west-1.amazonaws.com/NHATS_R"
                    + str(id)
                    + "_ClockDrawings/"
                    + value[0]
                    + ".tif"
                )

                # Open files and convert to work with Image in PIL
                response = requests.get(url)  # , stream = True)
                f = io.BytesIO(response.content)
                im_pil = Image.open(f)

                # Resize pil image files
                resized = im_pil.resize((im_pil.width // 16, im_pil.height // 16))
                imarray1 = np.array(resized)

                # imarray = np.logical_not(np.array(im)).astype(int) #bool to int, inverts values
                # image_data.append(get_coordinates(imarray1))  # , imarray1.shape[0], imarray1.shape[1]))

                # Store the np array images into a list
                store_images.append(imarray1)

                # Visualize the resized images
                viz_image(imarray1, resized)
                counter += 1

    return store_images


def viz_image(image, resized):
    print("shape: ", image.shape)

    # revert
    im2 = Image.fromarray(np.array(resized))
    plt.imshow(im2)
    plt.show()


def get_coordinates(data):  # , height, width):
    image = data
    image_array = []

    # for y in range(0, height):
    # for x in range(0, width):
    # if image[y][x] == False:
    # image_array.append((y, x))
    image_array = np.where(image == False)  # np.argwhere( image == False)
    return image_array

