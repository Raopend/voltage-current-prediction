import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec


def Show_clusters():
    # 加载模型
    kmeans_loaded = load("C:/Users/46444/student/1/biye/predict/voltage-current-prediction-release/voltage-current-prediction-release/notebook/model/kmeans_model.joblib")
    # 加载数据
    data = pd.read_csv('C:/Users/46444/student/1/biye/predict/voltage-current-prediction-release/voltage-current-prediction-release/data/dnaq_history_data_2022_ext2.csv')
    # 选择电流指标列
    electric_features = ['Ia', 'Ib', 'Ic']
    X = data[electric_features]
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 划分数据集
    split_point = int(len(X_scaled) * 0.8)
    # 获取训练集
    X_train = X_scaled[:split_point]
    # 使用加载的模型进行预测
    clusters = kmeans_loaded.predict(X_train)
    # 使用PCA降维到2维以便可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    # 聚类中心
    centers = pca.transform(kmeans_loaded.cluster_centers_)
    # 可视化聚类结果，为每个聚类绘制散点图并添加图例
    plt.figure(figsize=(12, 8))
    for cluster_number in range(kmeans_loaded.n_clusters):
        plt.scatter(X_pca[clusters == cluster_number, 0], X_pca[clusters == cluster_number, 1],
                    label=f'Cluster {cluster_number + 1}', marker='o', edgecolor='k', s=100, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5, marker='X', label='Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of clustered data with 2 clusters')
    plt.legend()
    plt.show()


# 异常检测
def detect_anomaly(new_data_point, cluster_centers, threshold=2.0):
    distances = [np.linalg.norm(center - new_data_point) for center in cluster_centers]
    min_distance = min(distances)
    is_anomaly = min_distance > threshold
    return is_anomaly, min_distance


# def insert_segments(original_data, new_data, segments=500):
#     # 确保新数据能够被平均分成指定的段数
#     assert len(new_data) % segments == 0

#     # 计算每段的长度
#     segment_length = len(new_data) // segments

#     # 获取原始数据的长度
#     original_length = len(original_data)

#     # 生成随机插入的位置
#     insert_positions = np.sort(np.random.choice(range(original_length + 1), segments, replace=False))

#     # 创建一个空的列表来存储新的数据集
#     new_dataset = []
#     last_pos = 0

#     # 对于每个插入位置，将原始数据和新数据段按顺序添加到new_dataset中
#     for pos in insert_positions:
#         # 添加当前位置之前的原始数据
#         new_dataset.append(original_data[last_pos:pos])
#         # 添加一个新数据段
#         start_new_segment = (pos % segments) * segment_length
#         new_dataset.append(new_data[start_new_segment:start_new_segment + segment_length])
#         last_pos = pos

#     # 添加剩余的原始数据
#     new_dataset.append(original_data[last_pos:])

#     # 将列表展平并转换为numpy数组
#     return np.vstack(new_dataset)


# 进行检测并获取检测数据
def Get_detect_data(datapath, model_path):
    # 加载模型
    kmeans_loaded = load(model_path)
    # 加载数据
    data = pd.read_csv(datapath)
    # 选择温度指标列
    electric_features = ['Ia', 'Ib', 'Ic']
    X = data[electric_features]
    # # 手动生成异常数据点
    # np.random.seed(42)
    # new_data_points = np.round(np.random.uniform(17, 20, size=(10000, 6)), 1)

    # # 将新数据点分成几段随机插入到X_test_original中
    # X_extended = insert_segments(X, new_data_points, segments=5000)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 划分数据集
    split_point = int(len(X_scaled) * 0.8)
    X_test = X_scaled[split_point:]
    # 对每个测试数据点进行异常检测
    anomaly_results = []
    distances = []
    # 反向映射到未标准化的原始数据点
    X_test_original = scaler.inverse_transform(X_test)

    for new_data_point in X_test:
        is_anomaly, distance = detect_anomaly(new_data_point, kmeans_loaded.cluster_centers_, threshold=2.0)
        anomaly_results.append((is_anomaly, distance))
        distances.append(distance)

    # 计算平均值和标准差
    mean_distance = np.mean(distances)
    std_dev_distance = np.std(distances)

    return X_test_original, distances, mean_distance, std_dev_distance


if __name__ == '__main__':
    # Show_clusters()
    # X_test_original, distances, mean, std = Get_detect_data()
    # print(X_test_original[:100])
    # print(distances[:100])
    data_path = 'C:/Users/46444/student/1/biye/predict/voltage-current-prediction-release/voltage-current-prediction-release/data/dnaq_history_data_2022_ext2.csv'
    model_path = 'C:/Users/46444/student/1/biye/predict/voltage-current-prediction-release/voltage-current-prediction-release/notebook/model/kmeans_model.joblib'
    X_test_original, distances, mean, std = Get_detect_data(data_path, model_path)
    print(len(X_test_original))
    pass
