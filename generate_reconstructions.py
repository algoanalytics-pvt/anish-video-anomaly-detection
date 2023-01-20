# internal imports
from suae import sUAE
import config
from natsort import natsorted
import os
import numpy as np
import cv2

source = ""
image_list = natsorted(os.listdir(source))
destination = ""


def reconstruct():
    suae = sUAE()
    suae.load()
    counter = 0
    for image_file in image_list:
        counter += 1
        print(counter)
        image = cv2.imread(os.path.join(source, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (224, 224),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_AREA,
        )
        array = np.expand_dims(np.array(image) / 255.0, axis=[0]).astype(np.float32)
        prediction = suae.model.predict(array)
        prediction = (prediction[0] * 255).astype(np.uint8)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        print(os.path.join(destination, image_file))
        cv2.imwrite(os.path.join(destination, image_file), prediction)


if __name__ == "__main__":
    reconstruct()
