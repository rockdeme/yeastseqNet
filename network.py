import tensorflow as tf
from tensorflow import keras


def threeconvnet(window_size_utr5=1500, window_size_orf=1500, window_size_utr3=1500, activation='relu',
                 initializer='he_normal'):
    """
    Convolutional network with 3 convolutional units channeled into a dense unit.
    :param initializer: weight initialization type
    :param activation: activation function
    :param window_size_utr5: maximum length of the 5'UTR in the examined transcriptome.
    :param window_size_orf: maximum length of the ORF in the examined transcriptome.
    :param window_size_utr3: maximum length of the 3'UTR in the examined transcriptome.
    :return: keras model object
    """
    if activation == 'lrelu':
        activation_function = tf.keras.layers.LeakyReLU(alpha=0.1)
    else:
        activation_function = activation

    # ORF codon sequence convolutional unit
    orf_input = keras.Input(shape=(64, window_size_orf, 1), name='orf')
    x = keras.layers.Conv2D(32, kernel_size=(64, 9), padding='valid', activation=activation_function,
                            kernel_initializer=initializer)(orf_input)
    x = keras.layers.MaxPooling2D(pool_size=(1, 30), padding='same')(x)
    x = keras.layers.Conv2D(128, kernel_size=(1, 9), padding='same', activation=activation_function,
                            kernel_initializer=initializer)(x)
    x = keras.layers.MaxPooling2D(pool_size=(1, 9))(x)
    orf_output = keras.layers.Flatten()(x)

    # UTR nucleotide sequence convolutional unit
    utr5_input = keras.Input(shape=(4, window_size_utr5, 1), name='utr5')
    x = keras.layers.Conv2D(32, kernel_size=(4, 36), padding='valid', activation=activation_function,
                            kernel_initializer=initializer)(utr5_input)
    x = keras.layers.MaxPooling2D(pool_size=(1, 30), padding='same')(x)
    x = keras.layers.Conv2D(128, kernel_size=(1, 36), padding='same', activation=activation_function,
                            kernel_initializer=initializer)(x)
    x = keras.layers.MaxPooling2D(pool_size=(1, 30))(x)
    utr5_output = keras.layers.Flatten()(x)

    # UTR nucleotide sequence convolutional unit
    utr3_input = keras.Input(shape=(4, window_size_utr3, 1), name='utr3')
    x = keras.layers.Conv2D(32, kernel_size=(4, 36), padding='valid', activation=activation_function,
                            kernel_initializer=initializer)(utr3_input)
    x = keras.layers.MaxPooling2D(pool_size=(1, 30), padding='same')(x)
    x = keras.layers.Conv2D(128, kernel_size=(1, 36), padding='same', activation=activation_function,
                            kernel_initializer=initializer)(x)
    x = keras.layers.MaxPooling2D(pool_size=(1, 30))(x)
    utr3_output = keras.layers.Flatten()(x)

    # numerical inputs
    sumtai = keras.Input(shape=(1,), name='sumtai', dtype="float32")
    gtai = keras.Input(shape=(1,), name='gtai', dtype="float32")
    sumcsc = keras.Input(shape=(1,), name='sumcsc', dtype="float32")
    gcsc = keras.Input(shape=(1,), name='gcsc', dtype="float32")
    sumtab = keras.Input(shape=(1,), name='sumtab', dtype="float32")
    gtab = keras.Input(shape=(1,), name='gtab', dtype="float32")
    residual = keras.Input(shape=(1,), name='residual', dtype="float32")
    # dense unit

    dense_input = keras.layers.Concatenate()([utr5_output,
                                              orf_output,
                                              utr3_output,
                                              sumtai,
                                              gtai,
                                              sumcsc,
                                              gcsc,
                                              sumtab,
                                              gtab,
                                              residual])
    x = keras.layers.Dense(64, activation=activation_function, kernel_initializer=initializer)(dense_input)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(2, activation=activation_function, kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.2)(x)
    dense_output = keras.layers.Dense(1, activation=activation_function)(x)
    model = keras.Model(inputs=[utr5_input, orf_input, utr3_input, sumtai, gtai, sumcsc, gcsc, sumtab, gtab, residual],
                        outputs=[dense_output])
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mean_squared_error'])

    return model
