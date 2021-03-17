"""
training a simple net on Chinese Characters classification dataset_prepared
we got about 90% accuracy by simply applying a simple CNN net

"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from alfred.dl.tf.common import mute_tf
mute_tf()
from alfred.utils.log import logger as logging
from dataset_prepared.casia_hwdb import load_ds, load_characters, load_val_ds
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import optimizers
from models.architecture.cnn_net import build_net_004


target_size = 96
num_classes = 3926
epoch_size = 25

# 是否调用上次模型
use_ckpt = True
# 是否调用SGD优化器
use_SGD = True

# 上次的模型的参数路径
last_model_path = './results/completed_model/1.1-epoch-25-lr-0.001-lr-0.01/Melnyk-25.h5'
# 最后模型参数保存路径
save_path = './results/completed_model/Melnyk-{epoch}.ckpt'
# 中途停止保存的模型路径
temporary_save_path = './results/temporary_model/Melnyk-{epoch}.ckpt'
# 结果保存路径
result_path = './results/data_results/'
#filepath = 'MelnykNet_model/model.{epoch:02d}-{val_loss:.2f}.hdf5'


def judge_path(ckpt_save_path, temporary_save_path, result_path):
    pos = ckpt_save_path.rfind('/')
    pos_temp = temporary_save_path.rfind('/')
    # 文件夹路径
    file_path = ckpt_save_path[: pos] + '/'
    file_temp_path = temporary_save_path[: pos_temp] + '/'

    # 创建模型保存路径
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file_temp_path):
        os.makedirs(file_temp_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    # original is 64x64, add a channel dim
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def plot():
    # 绘制并保存损失和准确性
    # plot and save the losses and accuracies:
    df = pd.read_csv(result_path + 'training_%s.log' % 'HCCR')

    # 绘制损失
    # plot the train and the validation cost:
    plt.plot(df['loss'], label='train_loss')
    plt.plot(df['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(result_path + '%s-loss.png' % 'HCCR')
    plt.show()

    plt.gcf().clear()  # clear the figure before plotting another  在绘制另一个图形之前先清除图形

    # plot the train and the validation accuracy: 绘制精度
    plt.plot(df['accuracy'], label='train_accuracy')
    plt.plot(df['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(result_path + '%s-accuracy.png' % 'HCCR')
    plt.show()


def train():
    all_characters = load_characters()
    num_classes = len(all_characters)
    logging.info('all characters: {}'.format(num_classes))
    train_dataset = load_ds()
    train_dataset = train_dataset.repeat().shuffle(100).map(preprocess).batch(128) ## batch(32)

    val_ds = load_val_ds()
    val_ds = val_ds.shuffle(100).repeat().map(preprocess).batch(128)  ## batch(32)

    # 初始化模型
    model = build_net_004((96, 96, 1), num_classes)  # (64, 64, 1)
    # model = build_net_003((96, 96, 1), num_classes)# (64, 64, 1)

    # 查看所有层
    model.summary()
    logging.info('model loaded.')

    start_epoch = 0
    # 会自动找到最近保存的模型文件
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(last_model_path))
    if latest_ckpt and use_ckpt:#
        # 获取上个模型的epoch数
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        # 加载权重参数
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('passing resume since weights not there. training from scratch')

    # 配置训练模型：优化器：Adam; 交叉熵损失函数; 模型评估标准：
    lr = 0.01
    if use_SGD:
        opt = optimizers.SGD(lr, momentum=0.9)  # SGD
    else:
        opt = optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, amsgrad=True)  # Adam

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # save model and weights
    callbacks = [
        # # 为True时，是只保存模型权重weight，为False时，保存整个模型，period表示保存模型的频率
        # ModelCheckpoint(temporary_save_path, monitor='val_accuracy', save_weights_only=False, verbose=0, mode='auto', period=10),
        ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, mode='auto', verbose=1, min_lr=0.001),
        CSVLogger(result_path + 'training_%s.log' % 'HCCR', append=False)  # epoch的训练结果
    ]

    try:
        model.fit(
            train_dataset,
            validation_data=val_ds,
            validation_steps=1830, # 验证集步数
            epochs=epoch_size,
            steps_per_epoch=7333, # 每个epoch的步数
            callbacks=callbacks)
        # 保存模型
        model.save_weights(save_path.format(epoch=epoch_size))
        model.save(os.path.join(os.path.dirname(save_path), 'Melnyk-25.h5'))
        logging.info('\nModel training is over.')

    except KeyboardInterrupt:  # 用户中断执行，ctrl + c
        model.save_weights(temporary_save_path.format(epoch='temp'))
        logging.info('\nInterrupt! keras model saved.')

    # 绘制训练集和验证集的损失值和精度值
    plot()


if __name__ == "__main__":
    # 判断文件路径是否存在，不存在则创建
    judge_path(save_path, temporary_save_path, result_path)
    train()
