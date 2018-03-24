import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('/home/deval/Desktop/face_exp/Reproduced_images/Training/1999.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

#Number of variations of the image to be generated
no_gen = 9


i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='1999_mod', save_prefix='3', save_format='jpeg'):
    i += 1
    if i > no_gen:
        break
        #break otherwise infinite images are been generated
