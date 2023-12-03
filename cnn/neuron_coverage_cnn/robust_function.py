import numpy as np


# 求两个向量之间的距离，p等于2的距离
def dist(array1, array2):
    """
    :param array1: 数组1
    :param array2: 数组2
    :return: 两个数组之间的距离
    """
    d = np.sqrt(np.sum(np.square(array1 - array2)))
    return d


def DSFCompute(embeddings_array, labels_array, k):
    """
    :param embeddings_array: 样本嵌入向量，二维array向量
    :param labels_array: 模型预测的样本标签，一维array向量
    :param labels_array: 分类数,整数值
    :return: DSF, FSA, FSD
    """
    # 创建标准化对象
    # scaler = MinMaxScaler()

    # 嵌入向量维数e_n
    e_n = embeddings_array.shape[1]
    # 求每一类的数据个数，保存到n中
    n = np.zeros(k, dtype=int)
    for element in labels_array:
        n[element] += 1

    # 定义center_list,由于内容均为一维向量，所以center_list使用二维np.array定义,并求center_list
    center_list = np.zeros((k, e_n), dtype=float)    # 一行代表一个均值向量，列数是均值向量个数，为k个
    count = 0
    for element in labels_array:
        center_list[element] = np.add(center_list[element], embeddings_array[count])  # 求每类的和
        count += 1
    # 求平均得到center_list
    for element in range(k):
        if n[element] != 0:
          center_list[element] /= n[element]

    # 求FSA_list和FSA，越接近1越好
    FSA_list = np.zeros(k, dtype=float)  # FSA是一个值，所以FSA_list用一维数组即可
    count = 0
    for element in labels_array:
        FSA_list[element] = FSA_list[element] + dist(center_list[element], embeddings_array[count])
        count += 1
    for element in range(k):
        if n[element] != 0:
            FSA_list[element] /= n[element]
    # 计算最小值和最大值
    min_FSA = np.min(FSA_list)
    max_FSA = np.max(FSA_list)
    # 使用 fit_transform 方法进行标准化
    FSA_list_norm = (FSA_list - min_FSA)/(max_FSA - min_FSA)
    # FSA_list_norm = scaler.fit_transform(FSA_list.reshape(-1, 1)).flatten()
    # print(FSA_list_norm)
    FSA = 1 - np.mean(FSA_list_norm)


    # 求FSD_list和FSD，越接近1越好
    FSD_list = np.zeros(int(k * (k-1) / 2), dtype=float)  # FSD是一个值，所以FSD_list用一维数组即可
    count = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            FSD_list[count] = dist(center_list[i], center_list[j])
            count += 1
    # 计算最小值和最大值
    min_FSD = np.min(FSD_list)
    max_FSD = np.max(FSD_list)
    # 使用 fit_transform 方法进行标准化
    FSD_list_norm = (FSD_list - min_FSD)/(max_FSD - min_FSD)
    # FSD_list_norm = scaler.fit_transform(FSD_list.reshape(-1, 1)).flatten()
    FSD = np.mean(FSD_list_norm)

    # 求DSF,越接近0越好
    DSF_list = np.zeros(int(k * (k - 1) / 2), dtype=float)  # DSF是一个值，所以DSF_list用一维数组即可
    count = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            DSF_list[count] = FSA_list[i] + FSA_list[j] - dist(center_list[i], center_list[j])
            count += 1
    # 计算最小值和最大值
    min_DSF = np.min(DSF_list)
    max_DSF = np.max(DSF_list)
    # 使用 fit_transform 方法进行标准化
    DSF_list_norm = (DSF_list - min_DSF)/(max_DSF - min_DSF)
    # DSF_list_norm = scaler.fit_transform(DSF_list.reshape(-1, 1)).flatten()
    DSF = np.mean(DSF_list_norm)
    return FSA, FSD, DSF

def GetModelOutput(model, second_model, data):
    d1 = model.predict(data)
    d2 = second_model.predict(data)
    return d1, d2