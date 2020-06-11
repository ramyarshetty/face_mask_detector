from keras.models import load_model

model=load_model('/Users/prajna/fd.h5')

from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 100)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

#img = cv2.imread('/Users/prajna/Desktop/hi.jpg')
img = image.load_img('/Users/prajna/Desktop/withmask.jpg', target_size=(224, 224))
img1 = cv2.imread('/Users/prajna/Desktop/withmask.jpg')
#img = img.convert('L')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
Y_pred = model.predict(images)
print(Y_pred)
(mask, withoutMask) = model.predict(images)[0]
classes = model.predict_classes(images)
label = "Mask" if classes==0 else "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
# include the probability in the label
label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
imag = cv2.putText(img1,label, org, font,  fontScale, color, thickness, cv2.LINE_AA)

# show the output image
imag = cv2.resize(imag, (600, 600))
cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Output", imag)
cv2.waitKey(3000)
cv2.destroyAllWindows()




img = image.load_img('/Users/prajna/Desktop/nomask.jpg', target_size=(224, 224))
img1 = cv2.imread('/Users/prajna/Desktop/nomask.jpg')
#img = img.convert('L')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
Y_pred = model.predict(images)
print(Y_pred)
(mask, withoutMask) = model.predict(images)[0]
classes = model.predict_classes(images)
label = "Mask" if classes==0 else "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
# include the probability in the label
label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
imag = cv2.putText(img1,label, org, font,  fontScale, color, thickness, cv2.LINE_AA)

# show the output image
imag = cv2.resize(imag, (600, 600))
cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Output", imag)
cv2.waitKey(3000)
cv2.destroyAllWindows()
