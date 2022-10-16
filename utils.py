import cv2

def read_masks(path, image_id):
    mask = cv2.imread(path + image_id)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask

def read_images(path, image_id):
    img = cv2.imread(path + image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_image_masks(image_id, images_path, masks_path):
    x = cv2.imread(images_path + image_id)
    image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(masks_path + image_id)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return image, mask