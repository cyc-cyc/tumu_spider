#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import subprocess
import time
'''
def run_spider(building_name, id):
    # 这里调用你的爬虫逻辑
    try:
        # 这是一个示例，实际的爬虫逻辑应在这里实现
        print(f"Running spider for {building_name} with ID {id}")
        # 假设爬虫成功
        messagebox.showinfo("成功", f"爬虫已成功运行：{building_name} (ID: {id})")
    except Exception as e:
        messagebox.showerror("错误", f"爬虫运行失败：{str(e)}")
'''

def run_spider(building_name,id):
        print(f"调用爬虫，传递建筑名称: {building_name}...")
        subprocess.run(['python', 'main.py', building_name,id])  # 调用 spider.py 并传递参数
        print(building_name,"爬虫已完成，等待 20 秒...")
        time.sleep(20)  # 等待 20 秒



def start_spider():
    building_name = entry_building_name.get()
    id = entry_id.get()
    if building_name and id:
        run_spider(building_name, str(id))
    else:
        messagebox.showwarning("警告", "请填写所有字段")

# 创建主窗口
root = tk.Tk()
root.title("爬虫 GUI")

# 创建输入框和标签
label_building_name = tk.Label(root, text="建筑名称:")
label_building_name.pack(pady=5)
entry_building_name = tk.Entry(root, width=20)
entry_building_name.pack(pady=5)

label_id = tk.Label(root, text="ID:")
label_id.pack(pady=5)
entry_id = tk.Entry(root, width=20)
entry_id.pack(pady=5)

# 创建启动爬虫按钮
btn_start = tk.Button(root, text="开始爬虫", command=start_spider)
btn_start.pack(pady=20)

# 运行主循环
root.mainloop()