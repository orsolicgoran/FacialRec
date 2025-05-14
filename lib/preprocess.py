import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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
		img = color_correct_img(img)
		img = normalize_img(img)
		return img
	else:
		return None


def process_folder(input_base_dir_path, output_base_dir_path):
	if not os.path.exists(output_base_dir_path):
		os.makedirs(output_base_dir_path, exist_ok=True)
  
  
	img_extension = ".jpg"
	for root, dirs, files in os.walk(input_base_dir_path):
		for filename in files:
			if filename.lower().endswith(img_extension):
				input_image_path = os.path.join(root, filename)

				relative_path = os.path.relpath(input_image_path, input_base_dir_path)
				output_image_path = os.path.join(output_base_dir_path, relative_path)
    
				output_image_folder = os.path.dirname(output_image_path)
				if not os.path.exists(output_image_folder):
					os.makedirs(output_image_folder, exist_ok=True)
     
				preprocessed_image = process_img(cv2.imread(input_image_path), 416)
				if preprocessed_image is not None:
					preprocessed_image = preprocessed_image * 255.0
					preprocessed_image = preprocessed_image.astype(np.uint8)
					preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR)
					try:
						cv2.imwrite(output_image_path, preprocessed_image)
					except Exception as e:
						pass					
				else:
					pass
				


if __name__ == "__main__":
	process_folder("data/WIDER_test/images", "processed_data/WIDER_test/images")
	process_folder("data/WIDER_train/images", "processed_data/WIDER_train/images")
	process_folder("data/WIDER_val/images", "processed_data/WIDER_val/images")