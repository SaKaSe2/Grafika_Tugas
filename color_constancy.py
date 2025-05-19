import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi baca dan tampilkan gambar dengan matplotlib
def show_image(title, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Gray-World Assumption
def gray_world(img):
    img = img.astype(np.float32)
    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    img[:,:,0] = np.clip(img[:,:,0] * (avg_gray / avg_b), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * (avg_gray / avg_g), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)

# 2. White-Patch Assumption
def white_patch(img):
    img = img.astype(np.float32)
    max_b = np.max(img[:,:,0])
    max_g = np.max(img[:,:,1])
    max_r = np.max(img[:,:,2])

    img[:,:,0] = np.clip(img[:,:,0] * (255.0 / max_b), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * (255.0 / max_g), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * (255.0 / max_r), 0, 255)
    return img.astype(np.uint8)

# 3. Simplified Retinex Algorithm
def retinex(img, sigma=30):
    img = img.astype(np.float32) + 1.0  # untuk log(0) avoid
    log_img = np.log(img)

    blur_b = cv2.GaussianBlur(log_img[:,:,0], (0,0), sigma)
    blur_g = cv2.GaussianBlur(log_img[:,:,1], (0,0), sigma)
    blur_r = cv2.GaussianBlur(log_img[:,:,2], (0,0), sigma)

    retinex_b = log_img[:,:,0] - blur_b
    retinex_g = log_img[:,:,1] - blur_g
    retinex_r = log_img[:,:,2] - blur_r

    retinex_img = np.stack([retinex_b, retinex_g, retinex_r], axis=2)
    retinex_img = np.exp(retinex_img) - 1.0

    # Normalisasi ke 0-255
    retinex_img = retinex_img - np.min(retinex_img)
    retinex_img = retinex_img / np.max(retinex_img) * 255
    return retinex_img.astype(np.uint8)

# 4. Shade of Grey Algorithm
def shade_of_grey(img, p=6):
    img = img.astype(np.float32)
    norm_b = np.power(np.mean(np.power(img[:,:,0], p)), 1/p)
    norm_g = np.power(np.mean(np.power(img[:,:,1], p)), 1/p)
    norm_r = np.power(np.mean(np.power(img[:,:,2], p)), 1/p)

    avg_norm = (norm_b + norm_g + norm_r) / 3

    img[:,:,0] = np.clip(img[:,:,0] * (avg_norm / norm_b), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * (avg_norm / norm_g), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * (avg_norm / norm_r), 0, 255)
    return img.astype(np.uint8)

# Load gambar
img_path = '3.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan, pastikan path sudah benar")

# Tampilkan gambar asli
show_image('Original Image', img)

# Terapkan tiap algoritma
img_gray_world = gray_world(img)
img_white_patch = white_patch(img)
img_retinex = retinex(img)
img_shade_of_grey = shade_of_grey(img)

# Tampilkan hasilnya
show_image('Gray-World Assumption', img_gray_world)
show_image('White-Patch Assumption', img_white_patch)
show_image('Retinex Algorithm', img_retinex)
show_image('Shade of Grey Algorithm', img_shade_of_grey)
