import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from PIL import Image




def get_images_test(id_rounds):
    counter = 0
    store_images = []
    coordinate_data = []

    for id, array in id_rounds.items():
        for value in array:
            if counter < 100:
                url = 'https://clockimages.s3.us-west-1.amazonaws.com/NHATS_R' + str(
                    id) + '_ClockDrawings/' + value[0] + '.tif'

                response = requests.get(url)  # , stream = True)
                f = io.BytesIO(response.content)
                im_pil = Image.open(f)
                imarray1 = np.array(im_pil)
                resized = imarray1.resize(207, 160)
                # imarray = np.logical_not(np.array(im)).astype(int) #bool to int, inverts values
                store_images.append(resized)
                #coordinate_data.append(get_coordinates(imarray1))  # , imarray1.shape[0], imarray1.shape[1]))
                viz_image(resized)
                counter += 1

    return store_images






def get_coordinates(data):#, height, width):
  image = data
  image_array = []

  #for y in range(0, height):
    #for x in range(0, width):
      #if image[y][x] == False:
        #image_array.append((y, x))
  image_array = np.where(image == False) #np.argwhere( image == False)
  return image_array






def viz_image(image):
  print("shape: ", image.shape)

  # revert
  im2 = Image.fromarray(image)
  plt.imshow(im2)
  plt.show()



