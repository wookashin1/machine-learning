import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)

MODEL_SAVE_PATH = 'model/model-{epoch:03d}.h5'


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    model = Sequential()

    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))

    
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # ugao volana

    model.compile(loss='mse', optimizer=Adam(learning_rate=args.learning_rate))

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    
   
    train_generator = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
    valid_generator = batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False)

    
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )


    model.fit(
        train_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.nb_epoch,
        validation_data=valid_generator,
        validation_steps=len(X_valid) // args.batch_size,
        callbacks=[checkpoint],
        verbose=1
    )


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',              dest='data_dir',          type=str,   default='data')
    parser.add_argument('-n', help='number of epochs',            dest='nb_epoch',          type=int,   default=20)
    parser.add_argument('-s', help='number of batches per epoch', dest='steps_per_epoch',   type=int,   default=100)
    parser.add_argument('-b', help='batch size',                  dest='batch_size',        type=int,   default=64)
    parser.add_argument('-l', help='learning rate',               dest='learning_rate',     type=float, default=1.0e-3)
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    args = parser.parse_args()

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

