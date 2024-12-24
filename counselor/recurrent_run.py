# main.py
import subprocess
import time
from scrapy import cmdline
import pandas as pd
import subprocess



def run_spider(building_name,id):
        print(f"调用爬虫，传递建筑名称: {building_name}...")
        subprocess.run(['python', 'main.py', building_name,id])  # 调用 spider.py 并传递参数
        print(building_name,"爬虫已完成，等待 20 秒...")
        time.sleep(20)  # 等待 20 秒

if __name__ == "__main__":
    # 读取 Excel 文件
    df = pd.read_excel('/nfs-data/spiderman/建筑立面&材料做法_09_30_074008(1).xlsx',sheet_name = "立面", engine='openpyxl')
    # 提取特定列，例如 'column_name'
    column_data = df['地址']
    # 打印该列的数据
    address_list = []
    for item in column_data:
        if pd.notna(item):
            address_list.append(item)
    print(column_data)
    column_data = df['建筑名称/业态']
    name_list = []
    for item in column_data:
        if pd.notna(item):
            name_list.append(item)
    print(address_list)
    print(name_list)
    print(len(address_list),len(name_list))
    
    
    
# 提取建筑名和地址
    building_names = df['建筑名称/业态'].dropna().tolist()
    addresses = df['地址'].dropna().tolist()

    # 打印提取到的数据
    print("建筑名列表:", building_names)
    print("地址列表:", addresses)

    id = 0
    # 遍历建筑名和地址进行分析并打印
    for building_name, address in zip(building_names, addresses):
        print(f"建筑名: {building_name}, 地址: {address}")  # 打印每个条目的建筑名和地址
        run_spider(building_name,str(id))
        run_spider(address,str(id))
        id += 1
    # input()
    # for item in address_list:
    #     # subprocess.run(['python3', '../../google.py', item])  # 调用 scraper.py
    #     run_spider(item,id)
    # for item in name_list:
    #     # subprocess.run(['python3', '../../google.py', item])  # 调用 scraper.py
    #     run_spider(item,id)
    
