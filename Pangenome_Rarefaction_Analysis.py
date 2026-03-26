import pandas as pd
import numpy as np
import random

def process_and_save(file_index):
    # 读取Excel文件
    df = pd.read_excel('1212VV_gene_presence_absence_min.xlsx', header=None)

    # 处理列名：强制转字符串，去空格
    df.columns = df.columns.astype(str).str.strip().str.replace(r'\s+', '', regex=True)

    # 获取基因名称（假设第一行是标题，从第二行开始是数据）
    gene_names = df.iloc[1:, 0].values

    # 获取菌株存在性矩阵 (NumPy 二维数组)
    # 假设第一列是基因名，之后全是菌株数据
    gene_presence_absence = df.iloc[1:, 1:].values 

    # 获取所有菌株名称（对应矩阵的列名）
    strain_names = df.columns[1:].tolist()
    num_total_strains = len(strain_names)

    # 存储已选菌株的索引（使用索引操作比使用名称匹配更快）
    all_indices = list(range(num_total_strains))
    random.shuffle(all_indices)  # 直接随机打乱索引顺序

    selected_indices = []
    results = []
    all_genes_set = set()

    # 逐步添加菌株并计算
    for i in range(num_total_strains):
        current_strain_idx = all_indices[i]
        selected_indices.append(current_strain_idx)
        
        strain_name = strain_names[current_strain_idx]
        # print(f"Selected strain: {strain_name} (Total selected: {i + 1})")

        # 获取当前菌株的数据
        strain_data = gene_presence_absence[:, current_strain_idx]

        # 更新【总基因集合】(Pan-genome): 并集
        # 获取当前菌株中存在（值为1）的基因索引
        current_gene_indices = np.where(strain_data == 1)[0]
        unique_genes_current = {gene_names[j] for j in current_gene_indices}
        all_genes_set.update(unique_genes_current)
        all_genes_count = len(all_genes_set)

        # 更新【共有基因数】(Core-genome): 交集
        # 核心逻辑修正：
        # 使用 np.all 对已选列进行行扫描，如果某一行在所有选择列中都为 1，则为共有基因
        selected_matrix = gene_presence_absence[:, selected_indices]
        shared_genes_mask = np.all(selected_matrix == 1, axis=1)
        shared_genes_count = np.sum(shared_genes_mask)

        # 将结果添加到列表
        results.append({
            'Number of Strains': i + 1,
            'Total Gene Count': all_genes_count,
            'Shared Gene Count': shared_genes_count
        })

        # 打印当前进度（可选）
        if (i + 1) % 10 == 0 or i == 0:
            print(f"File {file_index} - Strains: {i + 1}, Total: {all_genes_count}, Shared: {shared_genes_count}")

    # 保存结果
    result_df = pd.DataFrame(results)
    output_file = f'gene_counts_and_shared_genes_{file_index:03d}.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"Finished file {file_index}: {output_file}")


def main():
    # 执行 200 次随机抽样模拟
    for i in range(1, 201):
        process_and_save(i)

if __name__ == "__main__":
    main()