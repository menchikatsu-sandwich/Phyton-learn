import cv2
import numpy as np

# Baca gambar
image = cv2.imread("asszilla.png")  # Ganti dengan path gambar yang ingin diuji

# Definisi kernel (contoh: edge detection)
custom_kernel = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.float32)

# Terapkan filter ke gambar
filtered_image = cv2.filter2D(image, -1, custom_kernel)

# Tampilkan hasil
cv2.imshow("AssZilla", image)
cv2.imshow("AssNigga", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
