import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal# 初始化器，生成具有正态分布的张量。

MODEL_FILEPATH = 'MelnykNet_model/Melnyk-Net.hdf5'
# some simple architecture
def build_net_001(input_shape, n_classes):
    assert len(input_shape) == 3, 'only support 3 channels'
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
        padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


def build_net_002(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


# this model is converge in terms of chinese characters classification
# so simply is effective sometimes, adding a dense maybe model will be better?
def build_net_003(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),
        layers.Dense(n_classes, activation='softmax')
    ])
    # input_ = Input(shape=input_shape)
    # x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
    #                   padding='same', activation='relu')(input_)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    # x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    #
    # x = layers.Flatten()(x)
    # x = layers.Dense(n_classes, activation='softmax')(x)
    #
    # model = Model(inputs=input_shape, outputs=x)
    return model


# new model to compression
def build_net_004(input_shape=(96, 96, 1), num_classes = 3926, reg = 1e-3):
    # if global_average_type == 'GWAP':
    #     GlobalAveragePooling = GlobalWeightedAveragePooling2D(kernel_initializer='ones')
    # elif global_average_type == 'GWOAP':
    #     GlobalAveragePooling  = GlobalWeightedOutputAveragePooling2D(kernel_initializer='ones')
    # else:
    #     GlobalAveragePooling = layers.GlobalAveragePooling2D()

    # if use_pretrained_weights:
    #     if global_average_type == 'GWAP':
    #         if not os.path.exists(MODEL_FILEPATH):
    #             print("\nError: 'Melnyk-Net.hdf5' not found")
    #             print('Please, donwload the model and place it in the current.')
    #             print('URL: https://drive.google.com/open?id=1s8PQo7CKpOGdo-eXwtYeweY8-yjs7RYp')
    #             input('\npress Enter to exit')
    #             exit()
    #
    #         model = load_model(MODEL_FILEPATH,
    #                            custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})
    #         print("loaded the previous model\n")
    #         return model
    #     else:
    #         print('pretrained weights available only for melnyk-net with gwap')
    #         exit()

    # 权重初始化
    random_normal = RandomNormal(stddev=0.001, seed=1996)

    # input_ = Input(shape=input_shape)
    model = tf.keras.Sequential([
        # build a network model
        Conv2D(input_shape = input_shape, filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)), # he_normal正态分布初始化，参数均值为0且标准差为sqrt(2/fan_in)  fan_in为权重张量的扇入
        BatchNormalization(),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),

        Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),

        Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),

        Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(192, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),

        Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
        BatchNormalization(),
        Activation('relu'),

        layers.GlobalAveragePooling2D(),
        Dropout(0.5),

        Dense(units=num_classes, kernel_initializer=random_normal, activation='softmax',
                  kernel_regularizer=l2(reg), bias_regularizer=l2(reg)),
    ])

    return model

# some architecture wrapped into tf.keras.Model
class CNNNet(tf.keras.Model):

    def __init__(self):
        pass
