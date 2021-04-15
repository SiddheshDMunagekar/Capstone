import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

import streamlit as st
from tempfile import NamedTemporaryFile

from PIL import Image


animals=Image.open('animal_classifier.jpeg')


st.image(animals,use_column_width=True)
st.sidebar.image(Image.open('Panda_49.JPG'),use_column_width=True)
st.sidebar.info('This Application is developed by Siddhesh D. Munagekar to classify Pandas Image with rest of the other animal Images')

def run_image_classifer_app():
    data_gen_train = ImageDataGenerator(rescale=1 / 255.0)

    # Initializing common hyper parameter
    image_width = 148
    image_height = 148
    batch_size = 16
    epochs = 10
    buffer= st.file_uploader("Upload your Image here Image here .jpg")
    temp_file = NamedTemporaryFile(delete=False)
    if buffer is not None:
        temp_file.write(buffer.getvalue())
        #print(load_img(temp_file.name))

        pic=Image.open(temp_file.name)
        st.image(pic,use_column_width=True)

    # panda_image=load_img('/content/drive/MyDrive/Machine_learning_2/LAB3/Panda_img/train/Panda/Panda_49.JPG',target_size=(image_width,image_height))
        panda_image = load_img(temp_file.name,
                           target_size=(image_width, image_height))


    ##Converting image to array##
        panda_img_array = keras.preprocessing.image.img_to_array(panda_image)

    # make data fit model
        trainX = np.reshape(panda_img_array,(1, panda_img_array.shape[0], panda_img_array.shape[1], panda_img_array.shape[2]))


    #url = 'https://drive.google.com/file/d/14Yxmsh_qYVhRcQe41aMuAuMxgLIWjlun/view?usp=sharing'
    #model_path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

    #model_path='C:\\Users\\siddh\\PycharmProjects\\pythonProject3\\Capstone\\my_model'

        saved_model = tf.keras.models.load_model('my_model/my_model')

    ############    Model Prediction based on saved model in the folder ##########################################################



        for i in range(0, len(trainX)):

            image = trainX[i]
            image = np.expand_dims(image,
                               axis=0)  # Insert a new axis that will appear at the axis position in the expanded array shape as model requires 4d array of image

            pred_classes = saved_model.predict(image,
                                           batch_size=1)  # Saved model#Predicting the class based on input images keeping batchsize =1 as mentioned in Lab

            if pred_classes == 0:

                st.error(" This image is not of Panda")
            # Incrementing not panda counter if there is a correct prediction.



            else:

                st.success("This image is of Panda ")
            # Incrementing pandas Counter if there is a correct prediction

    else:
        st.info('Please upload Animal image above')


if __name__ == '__main__':
    run_image_classifer_app()







