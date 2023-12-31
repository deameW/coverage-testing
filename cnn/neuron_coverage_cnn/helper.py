import hashlib
import zlib

import numpy as np
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Model

DATA_DIR = "../data/"
MODEL_DIR = "../models/"
RESULT_DIR = "../coverage/"

MNIST = "mnist"
CIFAR = "cifar"
SVHN = "svhn"
JS = "js"

DATASET_NAMES = [MNIST, CIFAR, SVHN, JS]

BIM = "bim"
CW = "cw"
FGSM = "fgsm"
JSMA = "jsma"
PGD = "pgd"
APGD = "apgd"
DF = "deepfool"
NF = "newtonfool"
SA = "squareattack"
ST = "spatialtransformation"
ATTACK_NAMES = [APGD, BIM, CW, DF, FGSM, JSMA, NF, PGD, SA, ST]


# helper function
def get_layer_i_output(model, i, data):
    layer_model = K.function([model.layers[0].input], [model.layers[i].output])
    tmp = layer_model([data])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))
    return ret


# the data is in range(-.5, .5)
def load_data(dataset_name):
    assert dataset_name in DATASET_NAMES
    x_train = np.load(DATA_DIR + dataset_name + '/benign/x_train.npy', allow_pickle=True)
    y_train = np.load(DATA_DIR + dataset_name + '/benign/y_train.npy', allow_pickle=True)
    x_test = np.load(DATA_DIR + dataset_name + '/benign/x_test.npy', allow_pickle=True)
    y_test = np.load(DATA_DIR + dataset_name + '/benign/y_test.npy', allow_pickle=True)
    return x_train, y_train, x_test, y_test


def calculate_geometric_diversity(original_outputs, adv_outputs):
    total_geometric_diversity = 0.0
    neuron_num = 0

    combined_outputs = np.concatenate([original_outputs, adv_outputs], axis=0)
    distances = np.linalg.norm(combined_outputs - np.mean(combined_outputs, axis=0), axis=1)
    total_geometric_diversity = np.mean(distances)
    neuron_num = np.prod(original_outputs.shape[1:])
    geometric_diversity = total_geometric_diversity / neuron_num

    return geometric_diversity


ROUND_NUM = 0


class Coverage:
    def __init__(self, model, x_train, y_train, x_test, y_test, x_adv, run_round):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_adv = x_adv
        self.previous_result = 0  # 上一次的结果
        self.increment = 1  # 增量
        self.threshold = 20  # NCD阈值，可以根据具体情况调整
        self.pattern_set = set()
        self.testing_result = []
        self.nc_results = []  # 存储每次调用 NC 函数的结果
        self.run_round = run_round

    # find scale factors and min num
    def scale(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]
        factors = dict()
        for i in layers:
            begin, end = 0, batch
            max_num, min_num = np.NINF, np.inf
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                tmp = layer_output.max()
                max_num = tmp if tmp > max_num else max_num
                tmp = layer_output.min()
                min_num = tmp if tmp < min_num else min_num
                begin += batch
                end += batch
            factors[i] = (max_num - min_num, min_num)
        return factors

    # 1 neuron_coverage_cnn
    def NC(self, layers, threshold=0., batch=1024):
        factors = self.scale(layers, batch=batch)
        # Calculate the total neuron number of all the layers
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        activate_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            buckets = np.zeros(neurons).astype('bool')
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                # scale the layer output to (0, 1)
                layer_output -= factors[i][1]
                layer_output /= factors[i][0]
                col_max = np.max(layer_output, axis=0)
                begin += batch
                end += batch
                buckets[col_max > threshold] = True
            activate_num += np.sum(buckets)
        # print('NC:\t{:.3f} activate_num:\t{} neuron_num:\t{}'.format(activate_num / neuron_num, activate_num, neuron_num))
        return activate_num / neuron_num, activate_num, neuron_num

    # 2 k-multisection neuron coverage, neuron boundary coverage and strong activation neuron coverage
    def KMNC(self, layers, k=10, batch=1024):
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        covered_num = 0
        l_covered_num = 0
        u_covered_num = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            print(neurons)
            begin, end = 0, batch
            data_num = self.x_train.shape[0]

            neuron_max = np.full(neurons, np.NINF).astype('float')
            neuron_min = np.full(neurons, np.inf).astype('float')
            while begin < data_num:
                layer_output_train = get_layer_i_output(self.model, i, self.x_train[begin:end])
                batch_neuron_max = np.max(layer_output_train, axis=0)
                batch_neuron_min = np.min(layer_output_train, axis=0)
                neuron_max = np.maximum(batch_neuron_max, neuron_max)
                neuron_min = np.minimum(batch_neuron_min, neuron_min)
                begin += batch
                end += batch
            buckets = np.zeros((neurons, k + 2)).astype('bool')
            interval = (neuron_max - neuron_min) / k
            # print(interval[8], neuron_max[8], neuron_min[8])
            begin, end = 0, batch
            data_num = self.x_adv.shape[0]
            while begin < data_num:
                layer_output_adv = get_layer_i_output(self.model, i, self.x_adv[begin: end])
                layer_output_adv -= neuron_min
                layer_output_adv /= (interval + 10 ** (-100))
                layer_output_adv[layer_output_adv < 0.] = -1
                layer_output_adv[layer_output_adv >= k / 1.0] = k
                layer_output_adv = layer_output_adv.astype('int')
                # index 0 for lower, 1 to k for between, k + 1 for upper
                layer_output_adv = layer_output_adv + 1
                for j in range(neurons):
                    uniq = np.unique(layer_output_adv[:, j])
                    # print(layer_output_adv[:, j])
                    buckets[j, uniq] = True
                begin += batch
                end += batch
            covered_num += np.sum(buckets[:, 1:-1])
            u_covered_num += np.sum(buckets[:, -1])
            l_covered_num += np.sum(buckets[:, 0])
        print('KMNC:\t{:.3f} covered_num:\t{}'.format(covered_num / (neuron_num * k), covered_num))
        print(
            'NBC:\t{:.3f} l_covered_num:\t{}'.format((l_covered_num + u_covered_num) / (neuron_num * 2), l_covered_num))
        print('SNAC:\t{:.3f} u_covered_num:\t{}'.format(u_covered_num / neuron_num, u_covered_num))
        return covered_num / (neuron_num * k), (l_covered_num + u_covered_num) / (
                neuron_num * 2), u_covered_num / neuron_num, covered_num, l_covered_num, u_covered_num, neuron_num * k

    # 3 top-k neuron coverage
    def TKNC(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        pattern_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            pattern_set = set()
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                for j in range(topk.shape[0]):
                    pattern_set.add(tuple(topk[j]))
                begin += batch
                end += batch
            pattern_num += len(pattern_set)
        print(
            'TKNC:\t{:.3f} pattern_num:\t{} neuron_num:\t{}'.format(pattern_num / neuron_num, pattern_num, neuron_num))
        return pattern_num / neuron_num, pattern_num, neuron_num

    # 4 top-k neuron patterns
    def TKNP(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        def to_tuple(x):
            l = list()
            for row in x:
                l.append(tuple(row))
            return tuple(l)

        pattern_set = set()
        layer_num = len(layers)
        data_num = self.x_adv.shape[0]
        patterns = np.zeros((data_num, layer_num, k))
        layer_cnt = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                patterns[begin:end, layer_cnt, :] = topk
                begin += batch
                end += batch
            layer_cnt += 1

        for i in range(patterns.shape[0]):
            pattern_set.add(to_tuple(patterns[i]))
        pattern_num = len(pattern_set)
        print('TKNP:\t{:.3f}'.format(pattern_num))
        return pattern_num

    # 5 geometric diversity
    def GD(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]

        for i in layers:
            begin, end = 0, batch
            original_outputs = []

            while begin < self.x_train.shape[0]:
                original_outputs.append(get_layer_i_output(self.model, i, self.x_train[begin:end]))
                begin += batch
                end += batch

            original_outputs_concatenated = np.concatenate(original_outputs, axis=0)

            begin, end = 0, batch
            while begin < data_num:
                adv_outputs = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                geometric_diversity = calculate_geometric_diversity(original_outputs_concatenated, adv_outputs)
                self.pattern_set.add(geometric_diversity)
                begin += batch
                end += batch

        if self.run_round == 0:
            print('GD:\t{:.3f}'.format(0))
            return 0

        pattern_num = len(self.pattern_set)
        print('GD:\t{:.3f}'.format(pattern_num))
        # 存储结果到 testing_result
        return pattern_num

    def NCD(self):
        train_size = self.x_train.shape[0]
        adv_size = self.x_adv.shape[0]
        current_result = self.previous_result

        for i in range(train_size):
            for j in range(adv_size):
                distance = np.linalg.norm(self.x_train[i] - self.x_adv[j])

                # Check if the distance is below the threshold
                if distance < self.threshold:
                    current_result += self.increment

        print('NCD:\t{}'.format(current_result))
        self.previous_result = current_result
        return current_result

    def extract_features(self, input_data):
        print(self.model.layers[-2])
        # 创建用于提取特征的中间层模型
        last_dense_index = len(self.model.layers) - 2
        intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.layers[last_dense_index].output)

        # 使用中间层模型提取特征
        features = intermediate_layer_model.predict(input_data)

        return features

    def STD(self):
        # 提取特征并形成特征矩阵
        x_adv_features = self.extract_features(self.x_adv)

        # 标准化特征矩阵
        scaler = StandardScaler()
        x_adv_features_scaled = scaler.fit_transform(x_adv_features)

        # 计算特征矩阵的标准差
        feature_std = np.std(x_adv_features_scaled, axis=0)

        # 计算标准差的范数
        std_norm = np.linalg.norm(feature_std)

        print('STD:\t{:.3f}'.format(std_norm))
        return std_norm

    def all(self, layers, batch=100):
        self.NC(layers, batch=batch)
        self.KMNC(layers, batch=batch)
        self.TKNC(layers, batch=batch)
        self.TKNP(layers, batch=batch)
