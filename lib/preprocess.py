import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_and_pad(img, target_size, pad_color=(0,0,0)):
	if img is None:
		raise ValueError


	img_size = img.shape[:2]
	ratio = float(target_size) / max(img_size)

	new_size = tuple([int(x * ratio) for x in img_size])
	# switch from height,width to width,height for cv2
	new_size = (new_size[1], new_size[0])
	try:
		resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
	except Exception as e:
		print(f"Error resizing image: {e}")
		return None

	# Calculate image padding to fit target size
	delta_w = target_size - new_size[0]
	delta_h = target_size - new_size[1]
	top, bottom = delta_h // 2, delta_h - (delta_h // 2)
	left, right = delta_w // 2, delta_w - (delta_w // 2)

	color = list(pad_color)
	new_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
								cv2.BORDER_CONSTANT, value=color)

	return new_img


def normalize_img(img):
	if img is not None:
		return img.astype(np.float32) / 255.0
	else:
		return None


def color_correct_img(img):
	if img is not None:
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	else:
		return None


def process_img(img, target_size, pad_color=(0, 0, 0)):
	if img is not None:
		img = resize_and_pad(img, target_size, pad_color)
		img = normalize_img(img)
		img = color_correct_img(img)
		return img
	else:
		return None


def process_folder(directory):
	pass


if __name__ == "__main__":
	image_path = "/home/goran/Programming/Python/projects/zavrsni/data/WIDER_train/images/0--Parade/0_Parade_marchingband_1_100.jpg"
	image = cv2.imread(image_path)
	if image is not None:
		processed_image = process_img(image, 512)
		if processed_image is not None:
			plt.imshow(processed_image)	
			plt.show()
		else:
			print(f"Failed to process image")
	else:
		print(f"Could not load image")