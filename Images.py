import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from PIL import Image




def get_images_test():
    counter = 0
    store_images = []
    image_data = []

    for id, array in id_rounds.items():
        for value in array:
            if counter < 1:
                url = 'https://clockimages.s3.us-west-1.amazonaws.com/NHATS_R' + str(
                    id) + '_ClockDrawings/' + value + '.tif'

                response = requests.get(url)  # , stream = True)
                f = io.BytesIO(response.content)
                im_pil = Image.open(f)
                imarray1 = np.array(im_pil)
                # imarray = np.logical_not(np.array(im)).astype(int) #bool to int, inverts values
                # store_images.append(imarray1)
                image_data.append(get_coordinates(imarray1))  # , imarray1.shape[0], imarray1.shape[1]))
                viz_image(imarray1, im_pil)
                counter += 1

    return image_data






def get_coordinates(data):#, height, width):
  image = data
  image_array = []

  #for y in range(0, height):
    #for x in range(0, width):
      #if image[y][x] == False:
        #image_array.append((y, x))
  image_array = np.where(image == False) #np.argwhere( image == False)
  return image_array






def viz_image(image, image_pil):
  print("shape: ", image.shape)

  # revert
  im2 = Image.fromarray(np.array(image_pil))
  plt.imshow(im2)
  plt.show()



