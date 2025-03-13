import cv2

# Membaca gambar (Pastikan ada file gambar di folder yang sama)
img = cv2.imread('asszilla.png')

# Menampilkan gambar
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
