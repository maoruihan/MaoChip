# Choosing SNPs Using Feature Selection (Unsupervised Feature Selection Using Feature Similarity, FSFS)

import numpy as np
import joblib


# Read data from `.txt` file
def read_from_txt(filePath, head=True):
    f = open(filePath)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(str, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    # 去表头
    if head:
        data_list = data_list[1:]
    else:
        pass
    array_data = np.array(data_list)
    return array_data


# get SNPs by chromosomes:
def get_snp():
    print('Loading snp data...')
    SNPs = read_from_txt('../Application/data/CFJY_YY_LD.txt', head=True)
    snp_by_chromosome = [[] for _ in range(19)]
    for item in SNPs:
        chromosome = int(item[0])
        snp_by_chromosome[chromosome - 1].append(item)

    joblib.dump(snp_by_chromosome, '../Application/data/snp_by_chromosome.pkl')

    return snp_by_chromosome


# get SNP maps by chromosomes:
def get_map():
    print('Loading map data...')
    snp_map = read_from_txt('../Application/data/CFJY_YY_map.txt', head=False)
    snp_map_by_chromosome = [[] for _ in range(19)]
    for item in snp_map:
        chromosome = int(item[0])
        snp_map_by_chromosome[chromosome - 1].append(item[1])

    joblib.dump(snp_map_by_chromosome, '../Application/data/snp_map_by_chromosome.pkl')

    return snp_map_by_chromosome


# calculate similarity matrix
def calc_similarity(SNPs, snp_map):
    num_snp = len(snp_map)
    similarity = np.zeros((num_snp, num_snp))

    for snp in SNPs:
        row = snp_map.index(snp[2])
        column = snp_map.index(snp[5])
        r2 = float(snp[-1])
        similarity[row, column] = r2
        similarity[column, row] = r2

    return similarity


def calc_similarity_by_chromosome():
    print('Loading chromosome data...')
    snp_map_chromosome = joblib.load('../Application/data/snp_map_by_chromosome.pkl')
    snp_chromosome = joblib.load('../Application/data/snp_by_chromosome.pkl')
    print('Calculating similarity by chromosome...')
    similarity_by_chromosome = []

    for i in range(19):
        print(i, len(snp_map_chromosome[i]))
        similarity_matrix = calc_similarity(snp_chromosome[i], snp_map_chromosome[i])
        print(similarity_matrix.shape)
        similarity_by_chromosome.append(similarity_matrix)

    joblib.dump(similarity_by_chromosome, '../Application/data/similarity_by_chromosome.pkl')

    return similarity_by_chromosome


def FSFS(similarity, snp_map, chr_num, k=5):
    similarity_matrix = 1 - np.array(similarity)
    R_sets = np.array(snp_map)
    k_inner = k

    t = 0
    theta = 0

    while k_inner > 1:

        def compute_dk0():

            # k 近邻索引
            k_neighbors_index = []
            # 第 k 个近邻的索引和值
            dki_index = []
            dki_value = []
            # 对每个 Fi 寻找其第 k 个近邻的距离及其索引
            for i, fi in enumerate(similarity_matrix):

                # fi 的 k 近邻索引
                fi_k_neighbors_index = np.argpartition(fi, k_inner)[:k_inner]
                if len(fi_k_neighbors_index) == 0:
                    print(fi, fi_k_neighbors_index)

                # fi 的第 k 个近邻的索引和值
                fi_kth_index = 0
                fi_kth_value = fi[fi_k_neighbors_index[0]]
                for idx in fi_k_neighbors_index:
                    if fi[idx] >= fi_kth_value:
                        fi_kth_value = fi[idx]
                        fi_kth_index = idx

                k_neighbors_index.append(fi_k_neighbors_index)
                dki_index.append(fi_kth_index)
                dki_value.append(fi_kth_value)

            return dki_value[np.array(dki_value).argmin()], dki_index[np.array(dki_value).argmin()], k_neighbors_index[
                dki_index[np.array(dki_value).argmin()]]

        # 最小的 dki 即 dk0
        # f0 索引
        # f0 的 k 近邻索引
        dk0, f0, f0k = compute_dk0()

        # 删除 similarity 中 f0k 行和列
        similarity_matrix = np.delete(similarity_matrix, f0k, axis=0)
        similarity_matrix = np.delete(similarity_matrix, f0k, axis=1)
        # 移除 f0 的 k 最近邻
        R_sets = np.delete(R_sets, f0k)

        if t == 0:
            theta = dk0

        if k_inner > len(R_sets) - 1:
            k_inner = len(R_sets) - 1

        if k_inner <= 1:
            break

        while dk0 > theta:
            k_inner = k_inner - 1
            if k_inner == 1:
                break
            dk0, f0, f0k = compute_dk0()

        # 监控染色体编号、迭代次数、阈值、dk0、k、子集R的大小
        print(chr_num, t, theta, dk0, k_inner, len(R_sets))

        t += 1

    return R_sets, theta


# 先执行下面这三个函数，生成相关性矩阵并保存为 pkl 文件
# get_snp()
# get_map()
# calc_similarity_by_chromosome()
# 按染色体编号生成二维 list，每个 list 为同一染色体内的 snp

# 开始 FSFS
print('Loading data...')
sim_by_chr = joblib.load('../Application/data/similarity_by_chromosome.pkl')
snp_map_by_chr = joblib.load('../Application/data/snp_map_by_chromosome.pkl')

R_by_chr = []
theta_by_chr = []

# 可以对每个染色体取不同的 k 值
k_chr = [35] * 19  # 全部取 35

for j in range(19):
    print('FSFS...', j, len(snp_map_by_chr[j]))
    R_i, theta_i = FSFS(sim_by_chr[j], snp_map_by_chr[j], chr_num=j, k=k_chr[j])
    R_by_chr.append(R_i)
    theta_by_chr.append(theta_i)

# 输出各染色体子集大小 并写入文件
f_result = open('../Application/data/result_FSFS.txt', 'a')
for j in range(19):
    print('R', j, ':', len(R_by_chr[j]), theta_by_chr[j])
    for m in R_by_chr[j]:
        f_result.writelines(str(j + 1) + ' ' + m + '\n')
f_result.close()
