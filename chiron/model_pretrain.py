"""
Author: Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
"""
import os
import json
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import sys
import argparse
import pandas as pd
from scipy.ndimage import gaussian_filter

def CNN_model(lr, in_shape, drop1, drop2):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (4, 4), activation='relu', input_shape=in_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(drop1)) # 0.1
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(drop1))
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(drop2))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    return model

def CNN_model_1layer(lr=0.0001, drop=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (4, 4), activation='relu', input_shape=(31, 31, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(drop))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    return model

def load_mat(path, transform, thr=0.98):
    mat = np.load(path)
    max_val = np.quantile(mat, thr)
    mat = mat / (max_val + 1e-8)
    mat[mat > 1] = 1
    mat[mat < 0] = 0
    # print(path, mat.shape)

    if transform is not None:
        mat = gaussian_filter(mat, sigma=1)
    return mat


def generate_inputs(train_dir, tns=None, n_epoch=100, batch_size=10, train_ratio=0.75):
    np.random.seed(0)
    pos_path = train_dir + '/pos'
    neg_path1 = train_dir + '/neg1'
    pos_files, neg_files = [], []

    print(f"adding files from {train_dir}...", flush=True)

    for _, __, files in os.walk(pos_path):
        for file in files:
            if file.endswith('.npy'):
                pos_files.append(f'{pos_path}/{file}')
            else:
                print("Wrong filename ending")
    for _, __, files in os.walk(neg_path1):
        for file in files:
            if file.endswith('.npy'):
                neg_files.append(f'{neg_path1}/{file}')
            else:
                print("Wrong filename ending")

    print(f"Number of positive data: {len(pos_files)}", flush=True)
    print(f"Number of negative data: {len(neg_files)}", flush=True)
    # for _, __, files in os.walk(neg_path2):
    #     for file in files:
    #         if file.endswith('.npy'):
    #             neg_files.append(f'{neg_path2}/{file}')
    # for _, __, files in os.walk(neg_path3):
    #     for file in files[0:500]:
    #         if file.endswith('.npy'):
    #             neg_files.append(f'{neg_path3}/{file}')

    train_samples, train_labels = [], []
    test_samples, test_labels = [], []

    for i in range(len(pos_files)):
        if np.random.uniform() < train_ratio:
            train_samples.append(load_mat(pos_files[i], tns))
            train_labels.append([1, 0])
        else:
            test_samples.append(load_mat(pos_files[i], tns))
            test_labels.append([1, 0])
    for i in range(len(neg_files)):
        if np.random.uniform() < train_ratio:
            train_samples.append(load_mat(neg_files[i], tns))
            train_labels.append([0, 1])
        else:
            test_samples.append(load_mat(neg_files[i], tns))
            test_labels.append([0, 1])

    image_size = train_samples[0].shape[0]

    test_mats = np.array(test_samples).reshape([len(test_samples), image_size, image_size, 1])
    test_labels = np.array(test_labels)

    order = list(np.arange(len(train_samples)))
    for epoch in range(n_epoch):
        np.random.shuffle(order)
        for batch in range(len(order) // batch_size):
            if batch % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch}:', flush=True)
            indices = order[batch * batch_size: (batch + 1) * batch_size]
            train_mats_batch = np.array([train_samples[idx] for idx in indices]).reshape(
                [batch_size, image_size, image_size, 1])
            train_labels_batch = np.array([train_labels[idx] for idx in indices])
            # print(train_mats_batch.shape, train_labels_batch.shape)
            yield epoch, batch, train_mats_batch, train_labels_batch, test_mats, test_labels


if __name__ == '__main__':
    base_dir = '/home/varsh/LoopCaller/'
    parser = argparse.ArgumentParser()
    parser.add_argument('num_layer', type=int)
    parser.add_argument('batch_name', type=str)
    parser.add_argument('train_dir', type=str)

    parser.add_argument('-l', '--lr', type=float, default=0.0001)
    # only for a 1layer model, else it gets over-ridden
    parser.add_argument('-d', '--drop', type=float, default=0.5)
    parser.add_argument('-s', '--transform', type=str, default=None)
    # new for this version
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-b', '--batch', type=int, default=10)

    args = parser.parse_args()

    # required arguments
    nl = args.num_layer
    run_name = args.batch_name

    # optional arguments
    lr = args.lr
    drop = args.drop
    train_dir = args.train_dir
    n_epoch = args.epoch
    batch_size = args.batch

    # set parameters
    train_ratio = 0.8

    if nl == 1:
        model = CNN_model_1layer(lr, drop)
    else:
        model = CNN_model(0.0001, (31, 31, 1), 0.1, 0.2)

    log = {
        'TrainAccu': [], 'TrainMSE': [],
        'TestAccu': [], 'TestMSE': []
    }

    train_res = []
    train_wrong_mats = []
    train_wrong_labels = []
    test_wrong_mats = []
    test_wrong_labels = []
    # def generate_inputs(train_dir, tns=None, n_epoch=100, batch_size=10, train_ratio=0.75):
    for epoch, batch, train_mats_batch, train_labels_batch, test_mats, test_labels in generate_inputs(
            train_dir,tns=args.transform, n_epoch=n_epoch, batch_size=batch_size, train_ratio=train_ratio):
        # if epoch == 0 and batch == 0:
        #     print('Test Input')
        #     print(test_mats.shape)
        #     std = np.std(test_mats, axis=0).reshape((31, 31))
        #     print(std)
        if batch == 0:
            test_pred_res = model.predict(test_mats, verbose=0)
            test_pred_labels = np.array([[1, 0] if elm[0] > elm[1] else [0, 1] for elm in test_pred_res])
            diff = np.sum(np.abs(test_pred_labels - test_labels)) // 2
            test_accuracy = 1 - diff / len(test_pred_labels)
            # print('Test Prediction:')
            # print(test_pred_res)
            # print(np.std(test_pred_res, axis=0))
            if epoch == 0:
                train_accuracy = 0.
            else:
                train_accuracy = 1 - np.sum(train_res) / len(train_res)
                train_res = []
            mse1 = model.evaluate(train_mats_batch, train_labels_batch, batch_size=batch_size, verbose=0)
            mse2 = model.evaluate(test_mats, test_labels, batch_size=batch_size, verbose=0)
            print(f'Start of epoch {epoch}, Train accu: {train_accuracy}, Test accu: {test_accuracy}; ')
            print(f'Train MSE: {mse1}, Test MSE: {mse2}')
            log['TrainAccu'].append(train_accuracy)
            log['TrainMSE'].append(mse1)
            log['TestAccu'].append(test_accuracy)
            log['TestMSE'].append(mse2)

        model.train_on_batch(train_mats_batch, train_labels_batch)

        train_pred_res = model.predict(train_mats_batch, verbose=0)
        train_pred_labels = np.array([[1, 0] if elm[0] > elm[1] else [0, 1] for elm in train_pred_res])
        diff = np.sum(np.abs(train_pred_labels - train_labels_batch), axis=1) // 2
        train_res.extend(diff)
        if epoch == n_epoch - 1:
            for i in range(len(diff)):
                if diff[i] == 1:
                    train_wrong_mats.append(train_mats_batch[i])
                    train_wrong_labels.append(train_labels_batch[i])

    test_pred_res = model.predict(test_mats, verbose=0)
    test_pred_labels = np.array([[1, 0] if elm[0] > elm[1] else [0, 1] for elm in test_pred_res])
    diff = np.sum(np.abs(test_pred_labels - test_labels), axis=1) // 2
    test_accuracy = 1 - np.sum(diff) / len(test_pred_labels)
    for i in range(len(diff)):
        if diff[i] == 1:
            test_wrong_mats.append(test_mats[i])
            test_wrong_labels.append(test_labels[i])

    train_accuracy = 1 - np.sum(train_res) / len(train_res)
    train_res = []
    mse1 = model.evaluate(train_mats_batch, train_labels_batch, batch_size=batch_size, verbose=0)
    mse2 = model.evaluate(test_mats, test_labels, batch_size=batch_size, verbose=0)
    print(f'End of Training, Train accu: {train_accuracy}, Test accu: {test_accuracy}; ', flush=True)
    print(f'Train MSE: {mse1}, Test MSE: {mse2}')

    model.save_weights(f'{base_dir}{run_name}_model.h5')
    json.dump(log, open('log.json', 'w'))

    # Check wrong train and test samples
    print(f'{len(train_wrong_labels)} Wrong training samples')
    metric_dir = base_dir + 'metrics/'
    if not os.path.exists(metric_dir):
        os.mkdir(metric_dir)
    if not os.path.exists(metric_dir + run_name):
        os.mkdir(metric_dir + run_name)
    np.save(metric_dir + run_name + '/train_wrong_mats.npy', np.array(train_wrong_mats))
    np.savetxt(metric_dir + run_name + '/train_wrong_label.txt', np.array(train_wrong_labels))
    print(f'{len(test_wrong_labels)} Wrong testing samples')
    np.save(metric_dir + run_name + '/test_wrong_mats.npy', np.array(test_wrong_mats))
    np.savetxt(metric_dir + run_name + '/test_wrong_label.txt', np.array(test_wrong_labels))

    # validate

# Start of epoch 38, Train accu: 0.9783898305084746, Test accu: 0.9666666666666667;
# End of Training, Train accu: 0.9792372881355932, Test accu: 0.95;
# Train MSE: 0.009438648819923401, Test MSE: 0.08631205558776855
# 49 Wrong training samples
# 15 Wrong testing samples
