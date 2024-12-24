# 网络爬虫

**代码配置**

```
mkdir /path/to/your/folder

cd /path/to/your/folder

git clone https://github.com/cyc-cyc/tumu_spider.git
```

**环境配置**

```
conda create -n spider python=3.8

conda activate spider

pip install -r requirements.txt
```

**代理配置**

根据设备的条件自行配置代理

**运行指南**
```
cd counselor
python main.py building_name building_id
```

爬取的文件以json格式保存于本地目录中