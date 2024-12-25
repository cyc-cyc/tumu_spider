# main.py
import subprocess
import time
import argparse
from scrapy import cmdline
import pandas as pd
import subprocess
import numpy as np
import os
import json
import re
import difflib
import shutil
import os
import subprocess
from datetime import date
import mysql.connector
import json
import shutil
import os

def run_script_in_env(env_path, file_dir, save_dir, cuda_devices):
    """
    在指定的虚拟环境中运行 Python 脚本，并传递命令行参数。

    :param env_path: 虚拟环境的 Python 解释器路径
    :param file_dir: 输入文件目录
    :param save_dir: 输出保存目录
    :param cuda_devices: 可见的 CUDA 设备
    """
    command = [
        env_path, 'run.py',
        '--file_dir', file_dir,
        '--save_dir', save_dir,
        '--model_name',model_name,
        '--model_dir',model_dir   
    ]

    env = {'CUDA_VISIBLE_DEVICES': cuda_devices}

    try:
        result = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
        print("Output:", result.stdout)  # 打印标准输出
        print("Error:", result.stderr)    # 打印错误输出（如果有）
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        
def copy_folder(source, destination):
    try:
        # 检查源文件夹是否存在
        if not os.path.exists(source):
            print(f"源文件夹 '{source}' 不存在。")
            return
        
        # 如果目标文件夹存在，先删除它
        if os.path.exists(destination):
            shutil.rmtree(destination)
            # print(f"目标文件夹 '{destination}' 已被删除。")
        
        # 复制文件夹
        shutil.copytree(source, destination)
        # print(f"文件夹 '{source}' 已成功复制到 '{destination}'。")
    
    except Exception as e:
        print(f"复制文件夹时出错: {e}")
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # 解析 JSON 文件为 Python 数据结构
            return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print("文件不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生错误: {e}")

def delete_from_information_extraction(record_id, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 定义删除 SQL 语句
        delete_query = "DELETE FROM information_extraction WHERE id = %s"
        
        # 执行删除操作
        cursor.execute(delete_query, (record_id,))
        conn.commit()  # 提交事务

        if cursor.rowcount > 0:
            print("成功删除数据。")
        else:
            print("未找到符合条件的记录。")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()
def fetch_data_from_mysql(host, user, password, database):
    # 连接到 MySQL 数据库
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    # 执行查询
    cursor.execute("SELECT * FROM information_extraction")  # 替换为你的表名
    rows = cursor.fetchall()

    # 打印结果
    for row in rows:
        print(row)

    # 关闭连接
    cursor.close()
    conn.close()

def insert_into_information_extraction(data, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 定义插入 SQL 语句
        insert_query = """
        INSERT INTO information_extraction (
            id,
            building_name,
            detailed_address,
            neighborhood,
            street,
            building_function,
            structure_type,
            longitude,
            latitude,
            above_ground_floors,
            underground_floors,
            age,
            completion_year,
            compute_year,
            features,
            surrounding_transportation,
            surrounding_facilities,
            total_households,
            households_per_floor,
            square_meter_planning,
            layout_planning,
            management_mode,
            building_morphology
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        # 执行插入操作
        cursor.execute(insert_query, data)
        conn.commit()  # 提交事务

        print("成功插入数据。")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def fetch_column_names_from_table(table_name,host, user, password, database):
    try:
        # 连接到 MySQL 数据库
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 执行查询以获取列名
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()

        # 打印结果
        print(f"表 '{table_name}' 的列名:")
        for column in columns:
            print(column[0])  # 每个列信息是一个元组，取第一个元素为列名

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()
current_date = date.today()
def create_directory(directory_path):
    try:
        # 创建文件夹（如果文件夹已经存在，将不会抛出异常）
        os.makedirs(directory_path, exist_ok=True)
        print(f"文件夹 '{directory_path}' 已成功创建。")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

def run_command_in_conda_env(env_name, command):
    try:
        # 使用 conda run 执行命令
        result = subprocess.run(['conda', 'run', '-n', env_name] + command, check=True, text=True, capture_output=True)
        
        # 打印输出结果
        print("输出:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e}")
        print("错误输出:", e.stderr)


def run_bash_script(script_path):
    try:
        # 执行 Bash 脚本
        result = subprocess.run(['bash', script_path], check=True, text=True, capture_output=True)
        
        # 打印输出结果
        print("输出:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"执行脚本时出错: {e}")
        print("错误输出:", e.stderr)

def delete_folder(folder_path):
    try:
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 删除文件夹及其内容
            shutil.rmtree(folder_path)
            # print(f"文件夹 '{folder_path}' 已成功删除。")
        else:
            # print(f"文件夹 '{folder_path}' 不存在。")
            pass
    
    except Exception as e:
        print(f"删除文件夹时出错: {e}")


'''
土木项目真伪判别模块，包含:
1.多个来源查缺补漏 + 检查是否冲突 
2.规则判别
3.源文追溯
'''
import os
import json
import re
import difflib
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, TextStreamer
from transformers import AutoTokenizer
import torch

import os

def find_files_by_extension(folder_path, file_extension):
    """
    查找指定文件夹及其子文件夹中的所有特定类型文件。

    :param folder_path: 文件夹路径
    :param file_extension: 文件类型（如 '.txt' 或 '.json'）
    :return: 包含所有匹配文件路径的列表
    """
    matched_files = []  # 用于存储找到的文件路径
    for root, _, files in os.walk(folder_path):  # 遍历文件夹及其子文件夹
        for file in files:
            if file.endswith(file_extension):  # 检查文件扩展名
                matched_files.append(os.path.join(root, file))  # 拼接完整路径
    return matched_files

def merge(extract_folder_path):
    """
    多个来源查缺补漏+检查是否冲突
    """
    def merge_json(base_json, new_json):
        """
        合并两个 JSON 对象。如果某个字段在两个 JSON 中不同，合并为一个列表显示。
        """
        for key, value in new_json.items():
            value = str(value)  # 确保是字符串类型
            try:
                if key in base_json:
                    if '未提及' in base_json[key]:
                        # 如果 base_json 中的值为 "未提及"，直接替换
                        base_json[key] = value
                    elif '存在冲突' in base_json[key]: # 如果已经存在冲突，补充
                        base_json[key] = str(base_json[key]) + " / " + str(value) 
                    elif '未提及' in value:
                        continue
                    else: # 判断是否存在冲突
                        ratio = difflib.SequenceMatcher(None, base_json[key], value).ratio() # 通过相似度判断是否存在冲突
                        if ratio > 0.6:
                            continue
                        else: # 如果存在冲突，合并
                            base_json[key] = "存在冲突：" + str(base_json[key]) + " / " + str(value) 
                else:
                    # 如果 base_json 中没有该字段，直接添加
                    base_json[key] = value
            except:
                import ipdb; ipdb.set_trace()
        return base_json
    # final_json
    # 读取多个抽取json结果
    # 1. 互相补充
    # 2. 检查是否冲突
    final_json = {}
    files_list = find_files_by_extension(extract_folder_path, '.json')
    for file in files_list:
        with open(file, 'r', encoding='utf-8') as f:
            if final_json == {}:  # 如果 final_json 为空，直接加载第一个 JSON 文件
                    final_json = json.load(f)
            else:
                try:
                    current_json = json.load(f)  # 加载当前 JSON 文件
                    final_json = merge_json(final_json, current_json)  # 合并 JSON
                except:
                    import ipdb; ipdb.set_trace()
    return final_json
    # 在这里编写第一个函数的具体实现


def check(merge_json):
    """
    规则判别, 存在冲突怎么处理？
    """
    def extract_max_number(text):
        numbers = re.findall(r'\d+\.?\d*', str(text))  # 匹配整数和小数
        numbers = [float(num) for num in numbers]  # 转换为浮点数
        return max(numbers) if numbers else None  # 返回最大值

    # 替换标点符号为 '.'
    def normalize_date_format(date_text):
        normalized_date = re.sub(r'[^\d.]', '.', str(date_text))  # 替换非数字和点为点
        return normalized_date
    
    def handle_field_value(value, extract_function=None):
        """
        处理字段值，检查是否为'未提及'或'存在冲突'，并根据需要进行额外处理。
        :param value: 原始字段值
        :param extract_function: 用于处理正常值的函数（可选）
        :return: 处理后的字段值
        """
        value = str(value)  # 确保是字符串类型
        if "未提及" in value:
            return "未提及"
        elif "存在冲突" in value:
            return "存在冲突"
        elif extract_function:
            return extract_function(value)
        return value

    def data_process(df):
        year = handle_field_value(df["完工年代"], normalize_date_format)
        floor_up = handle_field_value(df["地上层数"], extract_max_number)
        h = handle_field_value(df["层高"], extract_max_number)
        
        return year, floor_up, h

    fields_to_extract = ["建筑名称", "地上层数", "层高", "完工年代"]
    # 提取所需的字段
    extracted_data = {field: merge_json.get(field, "未提及") for field in fields_to_extract}
    year, floor, h = data_process(extracted_data)
    # import ipdb; ipdb.set_trace()
    # 直接根据条件判断输出对应规则
    # 年代
    # 1999.2.21前
    if year == '未提及' or year == '存在冲突':
        rule = '年份信息缺少或存在冲突，无法判断'
    elif year < '1999.2.21': # 根据层数判断
        if floor == '未提及' or floor == '存在冲突':
            rule = '1999.2.21前的建筑，若层数小于10，则大概率为框架结构，小概率为框架-剪力墙；若层数大于10，则大概率为框架-剪力墙结构，小概率为框架结构。'
        elif floor < 10:
            rule = '1999.2.21前的建筑，若层数小于10，则大概率为框架结构，小概率为框架-剪力墙。'
        else:
            rule = '1999.2.21前的建筑，若层数大于10，则大概率为框架-剪力墙结构，小概率为框架结构。'
    else: # 根据房屋高度判断
        if h == '未提及' or h == '存在冲突':
            rule = '1999.2.21后的建筑，若层高小于20m，建筑材料为混凝土结构，则为框架结构；建筑材料为钢结构，则为框架结构(含阻尼器)。若层高大于等于20m且小于等于50m，建筑材料为混凝土结构，则为框架-剪力墙；建筑材料为钢结构，则为框架-支撑(含BRB、阻尼器)。若层高大于50m，建筑材料为混凝土结构，则为采用减(隔)震技术的框架-剪力墙；建筑材料为钢结构，则为采用减(隔)震技术的框架(含SRC)+支撑体系(BRB)。'
        elif h < 20:
            rule = '1999.2.21后的建筑，若层高小于20m，建筑材料为混凝土结构，则为框架结构；建筑材料为钢结构，则为框架结构(含阻尼器)。'
        elif h <= 50:
            rule = '1999.2.21后的建筑，若层高大于等于20m且小于等于50m，建筑材料为混凝土结构，则为框架-剪力墙；建筑材料为钢结构，则为框架-支撑(含BRB、阻尼器)。'
        else:
            rule = '1999.2.21后的建筑，若层高大于50m，建筑材料为混凝土结构，则为采用减(隔)震技术的框架-剪力墙；建筑材料为钢结构，则为采用减(隔)震技术的框架(含SRC)+支撑体系(BRB)。'
    # 保存在新的列中
    merge_json["规则"] = rule
    return merge_json # 返回更新后的json


def trace_llm(final_json, spider_folder_path, llm_model_path):
    """
    源文追溯，输出对应的句子。
    调用大模型？源文需要分段处理
    或者先把完工年代反处理一下，然后直接查找
    存在冲突的不处理
    """
    def init_llm(model_dir):
        model_name = "Qwen1.5-14B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True,)

        model_dtype_dict = {
            "Qwen1.5-7B-Chat": "auto",
            "Qwen1.5-14B-Chat": torch.float16,
            "Qwen1.5-MoE-A2.7B": "auto",  # 需要32GB显卡
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=model_dtype_dict[model_name],)
        prompt = "请帮我在以下段落中“{text}”，找到“{value}”所在的句子，请直接回答句子，如果找不到请回答“追溯失败”。"
        return model_name, model, tokenizer, prompt
    
    def merge_data(raw_data, MAX_L=2048, RED_L=100):
        # 按照最大长度(MAX_L)拆分数据, 每条数据的前面附带上一条数据的冗余(RED_L), 防止信息丢失
        if len(raw_data) <= MAX_L:
            return [raw_data]
        # 第一条数据前面不附带冗余, 直接取MAX_L长度
        data = [raw_data[:MAX_L]]
        # 有效数据长度
        ACT_L = MAX_L - RED_L
        
        for i in range(MAX_L, len(raw_data), ACT_L):
            # 从第二条数据开始, 每条数据前面附带上一条数据的冗余
            # 实际有效长度为ACT_L
            data.append(raw_data[i-RED_L:i+ACT_L])

        return data

    def find_value_in_text(value, spider_path, model, tokenizer, prompt):
        files_list = find_files_by_extension(spider_path, '.txt')
        for file in files_list:
            with open(file, 'r', encoding='utf-8') as f:
                source_text = f.read()
                # 分段处理
                data= merge_data(source_text, MAX_L=800, RED_L=100)
                for text in data:
                    content = prompt.format(text=text, value=value)
                    messages = [{"role": "user", "content": content}]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt")

                    generated_ids = model.generate(
                        model_inputs.input_ids.to('cuda'),
                        max_length=8192,
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                    trace_result = str(response[0])
                    if "追溯失败" not in trace_result:
                        return trace_result
        # 如果没找到
        return "追溯失败"

    # 读取大模型
    model_name, model, tokenizer, prompt = init_llm(llm_model_path)
    trace_json = {} # 初始化一个json保存溯源结果
    # 遍历 final_json 中的每个字段和值，查找源文中是否存在
    for key, value in final_json.items():
        if key == "规则":
            continue # 规则不需要追溯
        if "未提及" in value:
            trace_json[key] = "未提及，无需追溯"
        elif "存在冲突" in value:
            trace_json[key] = "存在冲突，无法追溯"
        else:
            result = find_value_in_text(value, spider_folder_path, model, tokenizer, prompt)
            trace_json[key] = result
    return trace_json
    

def save_final_json(final_json, trace_json, show_folder_path):
    """
    保存最终结果到文件
    """
    # 保存 final_json 到文件
    final_json_path = os.path.join(show_folder_path, "final.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
    # 保存 trace_json 到文件
    trace_json_path = os.path.join(show_folder_path, "trace.json")
    with open(trace_json_path, 'w', encoding='utf-8') as f:
        json.dump(trace_json, f, ensure_ascii=False, indent=4)
    print(f"已保存最终结果到文件：{final_json_path}")


def main(spider_path, extract_path, show_path, llm_model_path):
        merge_json = merge(extract_path)
        final_json= check(merge_json) # 在merge_json中加上规则
        trace_json = trace_llm(final_json, spider_path, llm_model_path)
        save_final_json(final_json, trace_json, show_path)

def run_spider(building_name,id,root):
        print(f"调用爬虫，传递建筑名称: {building_name}...")
        print(root)
        # input()
        subprocess.run(['python', 'main.py', building_name,id,root])  # 调用 spider.py 并传递参数
        print(building_name,"爬虫已完成，等待 10 秒防止ip封禁...")
        time.sleep(10)  # 等待 20 秒

if __name__ == "__main__":
    # 读取 Excel 文件
    parser = argparse.ArgumentParser(description="Run information extraction on a specified directory.")
    parser.add_argument('--root', type=str, required=True, help="The save directory")
    args = parser.parse_args()
    # 检查输入目录是否存在
    if not os.path.exists(args.root):
        print(f"Error: The directory {args.file_dir} does not exist.")
        sys.exit(1)
    root = args.root
    df = pd.read_excel('/nfs-data/spiderman/建筑立面&材料做法_09_30_074008(1).xlsx',sheet_name = "立面", engine='openpyxl')
    # 提取特定列，例如 'column_name'
    selected_columns = df[['建筑名称/业态', '地址']]
    selected_columns = selected_columns.dropna(axis=0, how="all") 
    # print(selected_columns)
    column_data = df['地址']
    # 打印该列的数据
    address_list = []
    for item in column_data:
        if pd.notna(item):
            address_list.append(item)
    # print(column_data)
    column_data = df['建筑名称/业态']
    name_list = []
    for item in column_data:
        if pd.notna(item):
            name_list.append(item)
    # print(address_list)
    # print(name_list)
    # print(len(address_list),len(name_list))
    

    folder_to_delete = root + '/content/temp/'  # 替换为要删除的文件夹路径
    delete_folder(folder_to_delete)

    folder_to_delete = root + '/content/'+str(current_date)+'/'    
    delete_folder(folder_to_delete)
    # 提取建筑名和地址
    building_names = df['建筑名称/业态'].dropna().tolist()
    addresses = df['地址'].dropna().tolist()

    # 打印提取到的数据
    # print("建筑名列表:", building_names)
    # print("地址列表:", addresses)


    
    for id,row in selected_columns.iterrows():
    # 遍历建筑名和地址进行分析并打印    
        building_name = row['建筑名称/业态']
        address=row['地址']
        id = 43
        building_name = building_names[id]
        address = addresses[id]

        id = input("请输入待爬取建筑物的唯一标识符：")
        address = input("请输入待爬取建筑物的名称：")
        


        print("*************************************")
        print("执行爬虫程序中.....")
        time.sleep(5) 

        run_spider(address,str(id),root)
        
        print("爬取结果已存至",root+'/content/temp/')

        print("*************************************")
        print("执行抽取程序中.....")
        time.sleep(5) 
        folder_to_delete = root+'/result/temp/'
        delete_folder(folder_to_delete)
        folder_to_create = root + '/result/temp/'  
        create_directory(folder_to_create)
        
        # env_name = 'extract_env' 
        # command_to_run = ['python', 'merge_check_trace.py']  
        # python run.py --file_dir "/nfs-data/spiderman/content/temp/" --save_dir "/nfs-data/spiderman/result/temp/" --CUDA_VISIBLE_DEVICES "3,4,5"
        # run_command_in_conda_env(env_name, command_to_run)
        
        run_script_in_env("extract_env",root+"/content/temp/", root+"/result/temp/", "Qwen1.5-14B-Chat","/nfs-data/zhengliwei/Projects/SHLP/LLMs/Qwen1.5-14B-Chat","3,4,5")
        # script_to_run = 'run.sh'  
        # try:
        #     run_bash_script(script_to_run)
        # except:
        #     copy_folder(root+"/result/2024-11-28/"+str(id),folder_to_create)
        print("信息抽取结果已存至",root+'/result/temp/')

        print("*************************************")
        print("执行真伪判别程序中.....")
        time.sleep(15) 
        folder_to_delete = root+'/show/temp/'
        delete_folder(folder_to_delete)
        folder_to_create = root+'/show/temp/'  
        create_directory(folder_to_create)
        spider_path = root+'/content/temp/'
        extract_path = root+'/result/temp/'
        show_path = root+'/show/temp/'
        llm_model_path = '/nfs-data/zhengliwei/Projects/SHLP/LLMs/Qwen1.5-14B-Chat'
        # main(spider_path, extract_path, show_path,llm_model_path)  # 执行主函数


        env_name = 'tumu_test1' 
        command_to_run = ['python', 'merge_check_trace.py']  
        run_command_in_conda_env(env_name, command_to_run)
        print("真伪判别结果已存至",show_path)
        print("*************************************")
        print("存储至数据库中.....")
        time.sleep(5) 
        # fetch_data_from_mysql('b101.guhk.cc', 'tumu', 'TJtumu', 'tumu')


        file_path = root+'/show/temp/final.json'  
        data = read_json_file(file_path)

        building_name="未提及"
        longitude_latitude="未提及"
        province_city_district="未提及"
        detailed_address="未提及"
        neighborhood="未提及" 
        street="未提及" 
        building_function="未提及"
        structure_type="未提及"
        longitude="未提及"
        latitude="未提及" 
        above_ground_floors="未提及"
        underground_floors="未提及" 
        age="未提及" 
        completion_year="未提及" 
        compute_year="未提及" 
        features="未提及" 
        surrounding_transportation="未提及" 
        surrounding_facilities="未提及" 
        total_households="未提及" 
        households_per_floor="未提及" 
        square_meter_planning="未提及" 
        layout_planning="未提及" 
        management_mode="未提及" 
        building_morphology="未提及"
        if data:
            if '建筑名称' in data:
                building_name = data['建筑名称'] 
            if '经纬度' in data:
                longitude_latitude = data['经纬度'] 
            if '省市区' in data:
                province_city_district = data['省市区'] 
            if '详细地址' in data:
                detailed_address = data['详细地址'] 
            if '街区' in data:
                neighborhood = data['街区'] 
            if '街道' in data:
                street = data['街道'] 
            if '建筑功能' in data:
                building_function = data['建筑功能'] 
            if '建筑结构' in data:
                structure_type = data['建筑结构'] 
            if '经度' in data:
                longitude = data['经度'] 
            if '纬度' in data:
                latitude = data['纬度'] 
            if '地上层数' in data:
                above_ground_floors = data['地上层数'] 
            if '地下层数' in data:
                underground_floors = data['地下层数'] 
            if '屋龄' in data:
                age = data['屋龄'] 
            if '完工年代' in data:
                completion_year = data['完工年代'] 
            if '算法断代' in data:
                compute_year = data['算法断代'] 
            if '特色' in data:
                features = data['特色'] 
            if '周边交通' in data:
                surrounding_transportation = data['周边交通'] 
            if '周边设施' in data:
                surrounding_facilities = data['周边设施'] 
            if '总户数' in data:
                total_households = data['总户数'] 
            if '同层户数' in data:
                households_per_floor = data['同层户数'] 
            if '坪数规划' in data:
                square_meter_planning = data['坪数规划'] 
            if '格局规划' in data:
                layout_planning = data['格局规划'] 
            if '管理方式' in data:
                management_mode = data['管理方式'] 
            if '建筑物形态' in data:
                building_morphology = data['建筑物形态'] 
        data_to_insert = (
            id,
            building_name, 
            # longitude_latitude, 
            # province_city_district, 
            detailed_address, 
            neighborhood, 
            street, 
            building_function, 
            structure_type, 
            longitude, 
            latitude, 
            above_ground_floors, 
            underground_floors, 
            age, 
            completion_year, 
            compute_year, 
            features, 
            surrounding_transportation, 
            surrounding_facilities, 
            total_households, 
            households_per_floor, 
            square_meter_planning, 
            layout_planning, 
            management_mode, 
            building_morphology
        )

        print(data_to_insert)
        # delete_from_information_extraction(58, 'b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
        insert_into_information_extraction(data_to_insert,'b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
        # fetch_data_from_mysql('b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
        input()