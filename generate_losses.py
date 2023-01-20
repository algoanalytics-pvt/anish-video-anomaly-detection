import cv2
import numpy as np
from natsort import natsorted
import os
import utils
import matplotlib.pyplot as plt

input_source = ""
reconstruction_source = ""

input_list = natsorted(os.listdir(input_source))
reconstruction_list = natsorted(os.listdir(reconstruction_source))

losses = []
counter = 0

for input_file, reconstruction_file in zip(input_list, reconstruction_list):
    counter += 1
    print(counter)
    input = cv2.imread(os.path.join(input_source, input_file))
    input = cv2.resize(input, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    input = np.expand_dims(np.array(input) / 255.0, axis=[0]).astype(np.float32)
    reconstruction = cv2.imread(
        os.path.join(reconstruction_source, reconstruction_file)
    )
    reconstruction = cv2.resize(
        reconstruction, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA
    )
    reconstruction = np.expand_dims(np.array(reconstruction) / 255.0, axis=[0]).astype(
        np.float32
    )
    anomaly_loss = utils.anomaly_loss(input, reconstruction)
    losses.append(anomaly_loss.numpy())

input_source = ""
reconstruction_source = ""

input_list = natsorted(os.listdir(input_source))
reconstruction_list = natsorted(os.listdir(reconstruction_source))

for input_file, reconstruction_file in zip(input_list, reconstruction_list):
    input = cv2.imread(os.path.join(input_source, input_file))
    input = cv2.resize(input, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    input = np.expand_dims(np.array(input) / 255.0, axis=[0]).astype(np.float32)
    reconstruction = cv2.imread(
        os.path.join(reconstruction_source, reconstruction_file)
    )
    reconstruction = cv2.resize(
        reconstruction, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA
    )
    reconstruction = np.expand_dims(np.array(reconstruction) / 255.0, axis=[0]).astype(
        np.float32
    )
    anomaly_loss = utils.anomaly_loss(input, reconstruction)
    losses.append(anomaly_loss.numpy())

print(losses)
fig = plt.figure(figsize=(10, 10))
plt.plot(losses)
ymin, ymax = plt.gca().get_ylim()
plt.vlines(counter, 0, ymax, color="r")
plt.show()
