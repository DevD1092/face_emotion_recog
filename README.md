# Repository details

Contains two folders, four python scripts, and a Summary_Phase_1 file. The CNN is implemented with keras (tensorflow as backend).

* Folder - Synthesized images -- Contains the example of images generated with augmentation (blur, rotation, flip, width and height shift).

* reproduce_img.py -- Script for reproducing the images in the dataset FER2013 for better visualization.

* img_synth.py -- Script for augmenting and storing the images in the training dataset of FER2013. Examples of the augmented images are contained in the Synthesized images folder.

* face_emo.py -- Script for training and testing the CNN on the images contained in the dataset FER2013.

* face_emo_syn.py -- Script for training and testing the CNN on the synthetic images from the dataset FER2013.

* Summary_Phase_1 -- Summary of the CNN architecture, parameter tuning and their impacts, and final accuracy results.

* Folder - semi -- Contains the source code files for semi-supervised learning implementation (using Ladder network) -- Folder details are as below.

* cnn_clean.py -- For saving the clean encoder model and weights.

* cnn_noisy.py -- For saving the noisy encoder model and weights.

* ladder_final.py -- Ladder network construction and soruce file for the same.

* model_cl_lrelu_bt_1cnn_ep_15.h5 -- Clean encoder saved model to be used for the ladder.py file.

* model_cl_lrelu_bt_1cnn_ep_15.json -- Clean encoder saved model weights to be used for the ladder.py file.

* model_ns_lrelu_bt_1cnn_ns_cnn_ep_15.h5 -- Noisy encoder model saved model to be used for the ladder.py file.

* model_ns_lrelu_bt_1cnn_ns_cnn_ep_15.json -- Noisy encoder model weights model to be used for the laddery.py file.

* unlabel_data.py -- Python script for segregating the dataset into labeled and unlabeled.


