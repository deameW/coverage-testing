import argparse
import logging
from datetime import datetime
import pytz
import warnings
import numpy as np
from PIL import Image
from tensorflow import keras
from helper import load_data
import tensorflow as tf
import os
import robust_function

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


## custom time zone for logger
def customTime(*args):
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()


logging.Formatter.converter = customTime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VERBOSE = False

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

def reshapeDataset(x_test):
    # 指定新的高度和宽度
    new_height = 128
    new_width = 128
    # resize x_test
    x_test_resized = []
    # 遍历 x_train 中的每个图像
    for image in x_test:
        # 创建一个 Pillow 图像对象
        pil_image = Image.fromarray(image)

        # 调整图像大小
        pil_image = pil_image.resize((new_width, new_height))

        # 将 Pillow 图像对象转换为 NumPy 数组
        resized_image = np.array(pil_image)

        # 将调整大小后的图像添加到列表
        x_test_resized.append(resized_image)
    # 将列表转换为 NumPy 数组
    x_test_resized = np.array(x_test_resized)
    # 数据预处理
    x_test_resized = x_test_resized / 255.0  # 归一化，将像素值缩放到 [0, 1]
    x_train = x_test_resized
    return x_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack for CNN')
    parser.add_argument(
        '--dataset', help="Model Architecture", type=str, default="mnist")
    parser.add_argument(
        '--model', help="Model Architecture", type=str, default="lenet5")
    parser.add_argument(
        '--attack', help="Adversarial examples", type=str, default="fgsm")

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    attack_name = args.attack

    ## Prepare directory for loading adversarial images and logging
    adv_dir = "{}{}/adv/{}/{}/".format(
        DATA_DIR, dataset_name, model_name, attack_name)
    cov_dir = "{}{}/adv/{}/{}/".format(
        RESULT_DIR, dataset_name, model_name, attack_name)

    ## Load benign images from mnist, cifar, or svhn
    x_train, y_train, x_test, y_test = load_data(dataset_name)
    x_adv_path = "{}x_test.npy".format(adv_dir)
    x_adv = np.load(x_adv_path)

    if dataset_name == "js":
        x_train = reshapeDataset(x_train)
        x_test = reshapeDataset(x_test)
        # x_adv = reshapeDataset(x_adv)
    ## Load keras pretrained model for the specific dataset
    model_path = "{}{}/{}.h5".format(MODEL_DIR,
                                     dataset_name, model_name)
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # 创建一个新模型，以倒数第二层的输出为输出
    second_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    k = len(y_train[0]) # class of the training data
    robust_result_path = os.path.join("./", "robust_result_bak.txt")
    with open(robust_result_path, "w+") as f:
        for data in (x_test, x_adv):
            prediction1, prediction2 = robust_function.GetModelOutput(model, second_model, data)
            print(prediction1.shape)
            print(prediction2.shape)
            output = np.argmax(prediction1, axis=1)
            FSA, FSD, DSF = robust_function.DSFCompute(prediction2, output, k)
            f.write('FSA: {}   \n'.format(FSA))
            f.write('FSD: {}   \n'.format(FSD))
            f.write('DSF: {}   \n'.format(DSF))
            f.write("\n------------------------------------------------------------------------------\n")
            print("FSA = " + str(FSA))
            print("FSD = " + str(FSD))
            print("DSF = " + str(DSF))
