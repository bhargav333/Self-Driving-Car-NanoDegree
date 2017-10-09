from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Cropping2D


def Simple(input_shape):
    """
    Super simple model just normaliztion and single layer
    :param input_shape: 
    :return: 
    """

    model = Sequential()

    #normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))

    return model


def LeNetKerasMSE(input_shape, dropout=.3 ):

    """
    Implement classic lenet architecture in keras for regression
    """

    model = Sequential()

    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    return model


def NvidiaNet(input_shape, dropout = .3):
    """
    Implement the Nvidia model for self driving cars
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    :param input_shape: shape of the images (r,w,c)
    :param dropout: 
    :return: 
    """

    model = Sequential()

    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((65, 25), (0, 0))))

    # Allow the model to choose the appropriate color space
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='linear'))

    return model

if __name__ == '__main__':
    ch, row, col = 3, 160, 320  # Original Image shape
    model = NvidiaNet((row,col,ch))
    print(model.summary())