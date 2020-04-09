import csv
import sqlite3
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas
import pandas as pd
from sklearn.cluster import KMeans
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']



def showData(data, key_category, key_attribute):

    fig, ax = plt.subplots(figsize=(10, 5))
    pos = set(data[key_category].to_numpy())
    plt.title(key_attribute, fontsize=20)
    for i in pos:
        categorical_data = data[data[key_category] == i]
        points_x = categorical_data[key_attribute].to_numpy()
        points_y = categorical_data[key_category].to_numpy()
        mean = categorical_data.mean()
        point_x = mean[key_attribute]
        ax.scatter(points_x, points_y, s=30, marker="o", label="sample point")
        ax.scatter(point_x, i, s=200, marker="^", label="mean point")
    saveFig(key_attribute)
    fig.show()

def saveFig(figname, path='./fig', suffix='png'):
    try:
        path_name = os.path.join(path, "{}.{}".format(figname,suffix))
        plt.savefig(path_name)
    except FileNotFoundError:
        try:
            os.mkdir(path)
        except FileExistsError:
            return
        plt.savefig(os.path.join(path, figname, suffix))

def startProcess():
    df = pd.read_csv("王者荣耀英雄数据.csv", encoding="gbk")
    summary = df.describe()
    df = df.replace(
        ['坦克', '战士', '刺客', '辅助', '射手', '法师'],
        [0, 1, 2, 3, 4, 5]
    )
    keys = list(df)[1:-4]
    for i in keys:
        showData(df, '主要定位', i)


def load_dataset(file_name):
    data_mat = []
    header = []
    with open(file_name) as fr:
        header = fr.readline()
        lines = fr.readlines()
    for line in lines:
        cur_line = line.strip().split(",")
        flt_line = list(map(lambda x:float(x) if x[-1] != '%' else float(x[:-1]), cur_line[1:-3]))
        data_mat.append(flt_line)
    return np.array(data_mat)

def csvToDb(filename, dbname):
    conn = sqlite3.connect(dbname)
    df = pandas.read_csv(filename, encoding='gbk')
    df.to_sql(dbname, conn, if_exists='append', index=False)


def dist_eclud(vecA, vecB):
    vec_square = []
    for element in vecA - vecB:
        element = element ** 2
        vec_square.append(element)
    return sum(vec_square) ** 0.5

def Kmeans(data_set, k):
    m = data_set.shape[0]
    print(m)
    cluster_assment = np.zeros((m, 2))
    centroids = rand_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf; min_index = -1
            for j in range(k):
                dist_ji = dist_eclud(centroids[j,:], data_set[i,:])
                if dist_ji < min_dist:
                    min_dist = dist_ji; min_index = j
            if cluster_assment[i,0] != min_index:
                cluster_changed = True
            cluster_assment[i,:] = min_index, min_dist**2
        for cent in range(k):
            pts_inclust = data_set[np.nonzero(list(map(lambda x:x==cent, cluster_assment[:,0])))]
            centroids[cent,:] = np.mean(pts_inclust, axis=0)
    return centroids, cluster_assment

def rand_cent(data_set, k):
    n = data_set.shape[1]
    centroids = np.zeros((k, n))
    for j in range(n):
        min_j = float(min(data_set[:,j]))
        range_j = float(max(data_set[:,j])) - min_j
        centroids[:,j] = (min_j + range_j * np.random.rand(k, 1))[:,0]
    return centroids

def ReadData(filename):
    with open(filename) as f:
        file = csv.reader(f)
        for row in file:
            print(row)


if __name__ == '__main__':
    startProcess()