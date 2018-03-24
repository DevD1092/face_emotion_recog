import os
import csv
import argparse
import numpy as np
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output directory")

args = parser.parse_args()

w, h = 48, 48

image = np.zeros((h, w), dtype=np.uint8)
id = 1
with open(args.file, 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = datareader.next()
    print headers 
    for row in datareader:  
        emotion = row[0]
        pixels = map(int, row[1].split())
        usage = row[2]
        #print emotion, type(pixels[0]), usage
        pixels_array = np.asarray(pixels)

        image = pixels_array.reshape(w, h)
        #print image.shape

        stacked_image = np.dstack((image,) * 3)
        #print stacked_image.shape


        image_folder = os.path.join(args.output, usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , str(id) + '.jpg')
        scipy.misc.imsave(image_file, stacked_image)
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
