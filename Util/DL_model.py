from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop

def NAAF_resnet(length=44, filters=32, kernel_size=5):
    if length % 4 != 0:
        raise ValueError("Length must be a multiple of 4")

    row_elements = length // 4

    input_layer = Input(shape=(length,))

    x = Reshape((4, row_elements))(input_layer)

    x = Permute((2, 1))(x)

    x1 = Conv1D(filters, kernel_size=1, padding='same', strides=1, activation='linear')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv1D(filters, kernel_size=kernel_size, padding='same', strides=1, activation='linear')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv1D(filters, kernel_size=kernel_size, padding='same', strides=1, activation='linear')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x = Add()([x1, x3])

    x = Flatten()(x)

    # first dense layer
    x = Dense(64, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second dense layer
    x = Dense(64, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['loss', 'acc'])

    return model


if __name__ == '__main__':
    # test code for model compling
    model = NAAF_resnet(length=40, filters=32, kernel_size=3)
    model.summary()