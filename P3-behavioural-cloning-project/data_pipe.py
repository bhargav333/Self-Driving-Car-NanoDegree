import glob
import os
import pandas as pd
import cv2
import numpy as np

SIDE_CORRECTION = .25


def load_samples_metadata(exclude_second_track=True , ds = 1):
    """
    Load some metadata for the training and validation samples,
    decide weather to keep or excluding the second track
    :return: 
    """

    if exclude_second_track:
        exclude_second_track = 'track2'

    # Train data

    train_samples = []
    for folder in sorted(glob.glob('data/*_data'), reverse=True):
        folder = os.path.abspath(folder)
        if exclude_second_track and exclude_second_track in folder: continue

        print('Loading TRAINING data from {}'.format(folder))
        samples_tmp = pd.read_csv('{}/driving_log.csv'.format(folder), encoding='utf8')
        for col in ['left', 'right', 'center']:
            samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: os.path.join(folder, x))

        train_samples.extend(samples_tmp.to_dict('records'))

    # Validation

    validation_samples = []
    for folder in sorted(glob.glob('data/validation*'), reverse=True):
        folder = os.path.abspath(folder)
        if exclude_second_track and exclude_second_track in folder: continue

        print('Loading VALIDATION data from {}'.format(folder))
        samples_tmp = pd.read_csv('{}/driving_log.csv'.format(folder), encoding='utf8')
        for col in ['left', 'right', 'center']:
            samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: os.path.join(folder, x))

        validation_samples.extend(samples_tmp.to_dict('records'))

    train_samples = np.array(train_samples)
    validation_samples = np.array(validation_samples)

    train_samples = train_samples[::ds]
    validation_samples = train_samples[::ds]

    # For the training samples augment the data:
    print('Train samples : {}'.format(len(train_samples)))
    print('Validation samples  : {}'.format(len(validation_samples)))

    return train_samples, validation_samples


# The following image processing images are taken from
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

def random_flip_image(img, angle, prob=.5):
    if  np.random.rand() < prob:
        img = np.fliplr(img)
        angle = - angle
    return img, angle


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    # tr_y = 40 * np.random.uniform() - 40 / 2
    tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    col, row = image.shape[:2]
    image_tr = cv2.warpAffine(image, Trans_M, (row, col))

    return image_tr, steer_ang


def augment_image_file_train(line_data):
    """
    Take a sample from the data (row from csv file in json format) and 
    return a tuple of (img, angle). The img is randomly augmented using 
    horizontal shifts, brightness adjustments and randomly adding shadows.
    These transformations are taken from 
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    :param line_data: 
    :return: 
    """

    i_lrc = np.random.randint(3)

    if (i_lrc == 0):
        path_file = line_data['left'].strip()
        shift_ang = SIDE_CORRECTION

    if (i_lrc == 1):
        path_file = line_data['center'].strip()
        shift_ang = 0.

    if (i_lrc == 2):
        path_file = line_data['right'].strip()
        shift_ang = - SIDE_CORRECTION

    y_steer = line_data['steering'] + shift_ang
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, y_steer = trans_image(image, y_steer, 100)
    image = augment_brightness_camera_images(image)
    image = add_random_shadow(image)

    image = np.array(image)

    image, y_steer = random_flip_image(image, y_steer)

    return image, y_steer


def train_generator(samples,
                    batch_size=128,
                    loop_forever=True,
                    keep_prob=1.0,
                    min_angle=-100000,
                    max_angle=+100000):
    """
    Allow keras to stream samples from disk for the training set. 
    Every sample is augemeted using image processing, 0 steering data can be downsampled.
    
    :param samples: 
    :param batch_size: 
    :param loop_forever: 
    :param keep_prob: 
    :param min_angle: 
    :param max_angle: 
    :return: 
    """
    num_samples = len(samples)
    while 1:
        images = []
        angles = []

        for _ in range(0, num_samples):

            ii = np.random.randint(0, num_samples)

            batch_sample = samples[ii]

            if batch_sample['steering'] == 0 and np.random.rand() <= 1 - keep_prob:
                continue



            img, angle = augment_image_file_train(batch_sample)

            # May remove angles which are two wide
            if angle < min_angle - 1e-3 or angle > max_angle + 1e-3:
                continue

            images.append(img)
            angles.append(angle)

            if len(angles) == batch_size:
                X_train = np.array(images)
                y_train = np.array(angles).squeeze()

                yield X_train, y_train

                images = []
                angles = []

        if not loop_forever: break


def validation_generator(samples,
                         batch_size=128,
                         loop_forever=True,
                         load_images= True):
    """
    Load samples from disk for validation set, does not do any data transformation
    :param samples: rows from validation csv files as json 
    :param batch_size: 
    :param loop_forever: wheather to stop after one run over the entire data
    :return: 
    """

    num_samples = len(samples)

    while 1:

        images = []
        angles = []

        # Do not actually need to shuffle for the validation data
        # samples = sklearn.utils.shuffle(samples)

        for ii in range(0, num_samples):

            batch_sample = samples[ii]

            angle = batch_sample['steering']
            if load_images:
                image = cv2.imread(batch_sample['center'].strip())
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images.append(image)
            angles.append(angle)

            if len(angles) == batch_size:
                X_train = np.array(images)
                y_train = np.array(angles).squeeze()

                yield X_train, y_train

                images = []
                angles = []

        if not loop_forever: break


def load_data(lapname):
    """
    Load a center images and angle of a single 
    :param lapname: 
    :return: 
    """

    samples_tmp = pd.read_csv('data/{}/driving_log.csv'.format(lapname), encoding='utf8')

    images = list()
    angles = list()
    filenames = list()
    for _, row in samples_tmp.iterrows():
        row['center'] = 'data/{}/{}'.format(lapname, row['center'])
        filenames.append(row['center'])
        center_image = cv2.imread(row['center'])
        images.append(center_image)
        center_angle = float(row['steering'])
        angles.append(center_angle)

    X_train = np.array(images)
    y_train = np.array(angles).squeeze()

    return X_train, filenames, y_train


def set_paths_new_data():
    """
    This function is used to set correct filepaths when new data is acquired
    :return: 
    """

    # Put files in better format

    for folder in sorted(glob.glob('data/*_data'), reverse=True) + sorted(glob.glob('data/validation_*'), reverse=True):
        try:
            folder = os.path.abspath(folder)
            filename = '{}/dr' \
                       'iving_log.csv'.format(folder)
            print('Loading data from {}'.format(folder))
            samples_tmp = pd.read_csv(filename, encoding='utf8')
            samples_tmp.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake',
                                   'speed']

            for col in ['left', 'right', 'center']:
                samples_tmp[col] = samples_tmp[col].str.strip().apply(lambda x: 'IMG/' + x.split('/')[-1])

            samples_tmp.to_csv(filename, index=False, encoding='utf8')

        except FileNotFoundError:
            pass


if __name__ == '__main__':
    set_paths_new_data()
