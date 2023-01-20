#external imports
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import mean, sqrt, square, constant
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import threading

#internal imports
import config

losses = [0] * (config.box_dims * config.box_dims)

def anomaly_loss(label, prediction):
    mse = tf.reduce_mean(mean(square(prediction - label), axis=[1,2,3]))
    ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(label, prediction, 1.0, filter_size=4))
    total_loss = 100*mse + ssim
    return total_loss

if(config.create_data_generators == True):
	if config.augmentations == True:
		train_datagen = ImageDataGenerator(
						rescale=1.0/255.0,
						brightness_range=[0.90, 1.15],
						zoom_range=0.2,
						width_shift_range=0.125,
						height_shift_range=0.125,
						shear_range=0.125,
						fill_mode="nearest"
						)
	else:
		train_datagen = ImageDataGenerator(rescale=1.0/255.0)

	train_generator = train_datagen.flow_from_directory(
		directory=config.train_dataset_folder,
		target_size=config.image_dimensions,
		color_mode=config.color_mode,
		batch_size=config.batch_size,
		class_mode=config.class_mode,
		shuffle=config.shuffle
		)

	validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

	validation_generator = validation_datagen.flow_from_directory(
		directory=config.validation_dataset_folder,
		target_size=config.image_dimensions,
		color_mode=config.color_mode,
		batch_size=config.batch_size,
		class_mode=config.class_mode,
		shuffle=config.shuffle
		)

	def fixed_generator(generator):
		for batch in generator:
			yield (batch, batch)

def get_single_box(centre, width, height):
	w = width/2
	h = height/2
	x = centre[0]
	y = centre[1]
	return([int(x-w), int(y-h), int(x+w), int(y+h)])

def get_boxes(w=224, h=224, box_dims=config.box_dims):
    boxes = []
    for i in range(1, box_dims*2, 2):
        for j in range(1, box_dims*2, 2):
            boxes.append(get_single_box(centre=(i*(w/(box_dims*2)), j*(h/(box_dims*2))), width=w/box_dims, height=h/box_dims))
    return boxes

def evaluate_boxes(original, reconstruction):
    losses = []
    boxes = get_boxes()
    original = (original * 255).astype(np.uint8)
    original = Image.fromarray(original)
    reconstruction = (reconstruction * 255).astype(np.uint8)
    reconstruction = Image.fromarray(reconstruction)
    for i in range(len(boxes)):
        box = boxes[i]
        original_crop = original.crop((box[0], box[1], box[2], box[3]))
        original_crop = np.expand_dims(np.array(original_crop).astype(np.float32)/255.0, axis=0)
        reconstruction_crop = reconstruction.crop((box[0], box[1], box[2], box[3]))
        reconstruction_crop = np.expand_dims(np.array(reconstruction_crop).astype(np.float32)/255.0, axis=0)
        loss = anomaly_loss(original_crop, reconstruction_crop)
        print(np.array(loss))
        losses.append(np.array(loss))
    return losses

def evaluate_box(box, original, reconstruction, index):
    original_crop = original.crop((box[0], box[1], box[2], box[3]))
    original_crop = np.expand_dims(np.array(original_crop).astype(np.float32)/255.0, axis=0)
    reconstruction_crop = reconstruction.crop((box[0], box[1], box[2], box[3]))
    reconstruction_crop = np.expand_dims(np.array(reconstruction_crop).astype(np.float32)/255.0, axis=0)
    loss = anomaly_loss(original_crop, reconstruction_crop)
    losses[index] = np.array(loss[0])

def evaluate_boxes_threaded(original, reconstruction):
    boxes = get_boxes()
    original = (original * 255).astype(np.uint8)
    original = Image.fromarray(original)
    reconstruction = (reconstruction * 255).astype(np.uint8)
    reconstruction = Image.fromarray(reconstruction)
    threads = [threading.Thread(target=evaluate_box, args=(boxes[index], original, reconstruction, index)) for index in range(len(boxes))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return losses

def put_box(image, box):
	image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
	return image

def put_grid(image, boxes):
    for box in boxes:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
    return image