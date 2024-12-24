# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import Selector
from items import ContentItem
from .myqueue import MyQueue
import time
from langconv import *
from .filter_words import filter_url
from bs4 import BeautifulSoup
from scrapy.pipelines.images import ImagesPipeline
from urllib.parse import urlparse, urljoin
import re
import json
import requests
import os
import requests
from bs4 import BeautifulSoup
import time 


import argparse

import shutil
import os

def copy_folder(source, destination):
    try:
        # 检查源文件夹是否存在
        if not os.path.exists(source):
            # print(f"源文件夹 '{source}' 不存在。")
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
# building_name = '台北小巨蛋'
# url = "https://zh.wikipedia.org/wiki/%E8%87%BA%E5%8C%97%E5%B0%8F%E5%B7%A8%E8%9B%8B"

def get_zh_top_related_pages(building_name, num_pages=1):
    try:
        # Construct the Wikipedia search URL
        search_url = f"https://zh.wikipedia.org/w/index.php?search=~ 台北市&title={building_name.replace(' ', '+')}"
        location="台北市"
        keyword = building_name.replace(' ', '+')
        search_url = f"https://zh.wikipedia.org/w/index.php?search={location}+{keyword}&title=Special:搜索&profile=advanced&fulltext=1&advancedSearch-current=%7B%22fields%22%3A%7B%22phrase%22%3A%22{location}+{keyword}%22%2C%22intitle%22%3A%22{keyword}%22%7D%7D&ns0=1"
        search_url = f"https://zh.wikipedia.org/w/index.php?search={location}+{keyword}+intitle%3A{keyword}&title=Special%3A%E6%90%9C%E7%B4%A2&profile=advanced&fulltext=1&advancedSearch-current=%7B%22fields%22%3A%7B%22phrase%22%3A%22{location}+{keyword}%22%2C%22intitle%22%3A%22{keyword}%22%7D%7D&ns0=1&searchToken=ev92lxdh10i3czhczwsqvg3un"
        # print(search_url)
        # Send a GET request to the search URL
        response = requests.get(search_url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        # print(response)
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # Find all search result titles and their corresponding URLs
        search_results = soup.find_all('div', class_='mw-search-result-heading')
        search_result_urls = [result.a['href'] for result in search_results if result.a]
        
        # Limit the number of search results to the specified number
        search_result_urls = search_result_urls[:num_pages]

        return search_result_urls
    except Exception as e:
        print(f": {e}")
        return []

def get_en_top_related_pages(building_name, num_pages=1):
    try:
        # Construct the Wikipedia search URL
        search_url = f"https://en.wikipedia.org/w/index.php?search=~ {building_name.replace(' ', '+')}"
        # Send a GET request to the search URL
        response = requests.get(search_url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        # print(response)
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # Find all search result titles and their corresponding URLs
        search_results = soup.find_all('div', class_='mw-search-result-heading')
        search_result_urls = [result.a['href'] for result in search_results if result.a]
        
        # Limit the number of search results to the specified number
        search_result_urls = search_result_urls[:num_pages]
        
        return search_result_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_housefun_top_related_page(building_name, num_pages=1):
    try:
        # Construct the Wikipedia search URL
        search_url = f"https://buy.housefun.com.tw/%E7%A4%BE%E5%8D%80?hd_Keyword={building_name.replace(' ', '+')}&hd_Sequence=Sequp&hd_SearchGroup=Group01&hd_PM=1&hd_Tab=1"
        # search_url = f"https://buy.housefun.com.tw/region/%E5%8F%B0%E5%8C%97%E5%B8%82_c/?kw={building_name.replace(' ', '+')}"
        # Send a GET request to the search URL
        # print(search_url)
        response = requests.get(search_url)
        
        response.raise_for_status()  # Raise an exception for non-200 status codes
        # print(response)
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # Find all search result titles and their corresponding URLs
        search_results = soup.find_all('section', class_='m-list-obj')
        search_result_urls = ["https://buy.housefun.com.tw"+result.a['href'] for result in search_results if result.a]
        # print("search_results",search_results)
        # Limit the number of search results to the specified number
        search_result_urls = search_result_urls[:num_pages]
        
        return search_result_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
    

def get_houseprice_top_related_page(building_name, num_pages=1):
    try:
        cookies = {
            '__ltm_https_flag': 'true',
            '_ga': 'GA1.1.129916505.1714984941',
            '__ltmwga': 'utmcsr=(direct)|utmcmd=(none)',
            '_clck': '1nal7rn%7C2%7Cfme%7C0%7C1587',
            '_userid': '8292b389-8f27-4626-ba0a-07a3fdcd58cf',
            '__lt__cid': '163191e3-e155-4f0e-ae2c-9bf82e76e26e',
            '__referer': 'https://buy.houseprice.tw/',
            '_ga_95JXSFYLYP': 'GS1.1.1718547807.9.0.1718547920.60.0.0',
            '_gcl_au': '1.1.1175314618.1728439793',
            '_fbp': 'fb.1.1728439794373.721668710141328431',
            'hpwebmobile': '0',
            '__lt__sid': '2b7974e9-819660b1',
            '_pk_ref.28.090a': '%5B%22%22%2C%22%22%2C1728989812%2C%22https%3A%2F%2Fwww.houseprice.tw%2F%22%5D',
            '_pk_ses.28.090a': '*',
            '_ga_S49XNJ5E0Y': 'GS1.1.1728989785.13.1.1728989823.22.0.0',
            '_ga_BCVL7YCK7L': 'GS1.1.1728989785.4.1.1728989823.22.0.0',
            '_pk_id.28.090a': '0812ffd19c0cfefc.1717677192.4.1728989824.1728989794',
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
        }
        # Construct the Wikipedia search URL
        search_url = f"https://buy.houseprice.tw/list/%E5%8F%B0%E5%8C%97%E5%B8%82_city/{building_name.replace(' ', '+')}_kw/"
        # Send a GET request to the search URL
        response = requests.get(search_url)

        # response.raise_for_status()  # Raise an exception for non-200 status codes
        # print(response)
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, 'html.parser')

        # print(soup)
        # Find all search result titles and their corresponding URLs
        search_results = soup.find_all('section', class_='m-list-obj')
        search_result_urls = ["https://buy.houseprice.tw/"+result.a['href'] for result in search_results if result.a]
        # print("search_results",search_results)
        # Limit the number of search results to the specified number
        search_result_urls = search_result_urls[:num_pages]
        
        return search_result_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def get_sinyi_top_related_page(building_name, num_pages=1):
    try:
        # Construct the Wikipedia search URL
        search_url = f"https://www.sinyi.com.tw/communitylist/{building_name.replace(' ', '+')}-keyword/Taipei-city/100-103-104-105-106-108-110-111-112-114-115-116-zip/hotdeal-desc/index"
        # print(search_url)
        # Send a GET request to the search URL
        response = requests.get(search_url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        # print(response)
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # Find all search result titles and their corresponding URLs
        search_results = soup.find_all('div', id='communitySearchListContent')
        search_result_urls = ["https://www.sinyi.com.tw/"+result.a['href'] for result in search_results if result.a]
        # print("search_results",search_results)
        # Limit the number of search results to the specified number
        # print(search_result_urls)
        search_result_urls = search_result_urls[:num_pages]
        
        return search_result_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

def search_zh_urls(building_name,num_pages):
    # Building name for which you want to find related Wikipedia pages
    
    
    # Get the top 5 Wikipedia pages most relevant to the building
    top_related_pages = get_zh_top_related_pages(building_name, num_pages=num_pages)
    # Print the URLs of the top related pages
    if top_related_pages:
        for i, page_url in enumerate(top_related_pages, start=1):
            top_related_pages[i-1] = "https://zh.wikipedia.org/" + top_related_pages[i-1]
            # print(f"Page {i}: {page_url}")
        return top_related_pages
    else:
        
        # print("No related pages found.")
        return [""]
    

def search_en_urls(building_name,num_pages):
    # Building name for which you want to find related Wikipedia pages
    
    
    # Get the top 5 Wikipedia pages most relevant to the building
    top_related_pages = get_en_top_related_pages(building_name, num_pages=num_pages)
    # Print the URLs of the top related pages
    if top_related_pages:
        for i, page_url in enumerate(top_related_pages, start=1):
            top_related_pages[i-1] = "https://en.wikipedia.org/" + top_related_pages[i-1]
            # print(f"Page {i}: {page_url}")
        return top_related_pages
    else:
        
        print("No related pages found.")
        return []
        

def extract_main_content(html):
    soup = BeautifulSoup(html, 'html.parser')

    # 删除脚本和样式标签
    for script in soup(["script", "style"]):
        script.extract()

    # 获取所有文本内容
    texts = soup.get_text()

    # 删除多余空白字符和换行符
    cleaned_text = re.sub(r'\s+', ' ', texts).strip()

    return cleaned_text
def extract_chinese_from_html(html):
    # soup = BeautifulSoup(html, 'html.parser')
    # texts = soup.find_all(text=True)

    # chinese_and_digit_texts = []
    # for text in texts:
    #     # 使用正则表达式匹配汉字和数字
    #     chinese_and_digits = re.findall(r'[\u4e00-\u9fff0-9]+', text)
    #     for item in chinese_and_digits:
    #         chinese_and_digit_texts.append(item)

    # return chinese_and_digit_texts
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.find_all(text=True)
    chinese_texts = []
    for text in texts:
        # 使用正则表达式匹配汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        chinese_text = ''.join(chinese_chars)
        if chinese_text:
            chinese_texts.append(chinese_text)

    return chinese_texts

class ImageItem(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()

        
    def download_image(url, save_dir):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # 提取文件名
            filename = url.split('/')[-1]
            save_path = os.path.join(save_dir, filename)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"图片已保存：{save_path}")
        else:
            print("图片下载失败")
def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    # sentence = Converter('zh-hans').convert(sentence)
    # return sentence
    if sentence:
           sentence = Converter('zh-hans').convert(sentence)
           return sentence
    else:
        return sentence

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def split(url_list):
    '''
    分离两种不同的请求类型（分类/内容）
    :return:
    '''
    cates_url, content_url = [], []
    for url in url_list:
        if 'Category:' in url:
            cates_url.append(url)
        else:
            content_url.append(url)
    return cates_url, content_url


def filter(url):
    # 如果字符串url中包含要过滤的词，则为True
    filter_url = ['游戏', '幻想', '我的世界', '魔兽']
    for i in filter_url:
        if i in url:
            return True
    return False


class WiKiSpider(scrapy.Spider):
    urlQueue = MyQueue()
    name = 'wikipieda_spider'
    allowed_domains = ["zh.wikipedia.org", "en.wikipedia.org", "housefun.com.tw", "houseprice.tw", "sinyi.com.tw"]

    def __init__(self, building_name=None,id=None, *args, **kwargs):
        super(WiKiSpider, self).__init__(*args, **kwargs)
        self.id = id
        self.building_name = building_name.strip('"')
        # start_urls = ['https://buy.housefun.com.tw/Building/building_photo.aspx?bid=26212']
        # start_urls = ['https://zh.wikipedia.org/wiki/%E8%87%BA%E5%8C%97%E5%B8%82%E4%BF%A1%E7%BE%A9%E5%8D%80%E5%85%89%E5%BE%A9%E5%9C%8B%E6%B0%91%E5%B0%8F%E5%AD%B8']
        # input()
        self.start_urls = search_zh_urls(building_name,num_pages = 1) \
        + get_housefun_top_related_page(building_name,num_pages = 3)  \
                + get_houseprice_top_related_page(building_name,num_pages = 3) \
                    # + get_sinyi_top_related_page(building_name,num_pages = 5)
        if self.start_urls and self.start_urls[0] == '':
            self.start_urls.pop(0)
        # print(self.start_urls)
        # input()
        # print("start_urls",self.start_urls)
        # input()
    custom_settings = {
        'ITEM_PIPELINES': {'counselor.pipelines.WikiPipeline': 800}
    }

    def save_image(self, response):
        # 获取图片名称
        image_name = response.url.split('/')[-1]
        # 保存图片
        dir = "/nfs-data/spiderman/picture/"+self.building_name
        os.makedirs(dir,exist_ok=True)
        with open(dir + image_name, 'wb') as f:
            f.write(response.body)

    def add_scheme(self, url, base_url):
        
        parsed_url = urlparse(url)
        
        if not parsed_url.scheme:
            base_parsed_url = urlparse(base_url)
            url_with_scheme = urljoin(base_parsed_url.scheme + "://" + base_parsed_url.netloc, url)
            return url_with_scheme
        return url

    # scrapy默认启动的用于处理start_urls的方法
    def parse(self, response):
        '''
        在维基百科中，页面有两种类型，分别是分类页面，链接中包含Category，否则是百科页面，例如：
        分类页面：https://zh.wikipedia.org/wiki/Category:计算机科学
        百科页面：https://zh.wikipedia.org/wiki/计算机科学
        本方法用于对请求的链接进行处理，如果是分类型的请求，则交给函数1处理，否则交给函数2处理
        :param response: 候选列表中的某个请求
        :return:
        '''
        # 获得一个新请求
        this_url = response.url
        # self.urlQueue.delete_candidate(this_url)
        # self.start_urls = self.urlQueue.candidates
        # 说明该请求时一个分类
        # print('this_url=', this_url)
        self.urlQueue.load_npy()
        if 'Category:' in this_url:
            yield scrapy.Request(this_url, callback=self.parse_category, dont_filter=True)
        else:
            yield scrapy.Request(this_url, callback=self.parse_content, dont_filter=True)


    def parse_category(self, response):
        '''
        处理分类页面的请求
        :param response:
        :return:
        '''
        counselor_item = ContentItem()
        sel = Selector(response)
        this_url = response.url
        self.urlQueue.delete_candidate(this_url)
        search = sel.xpath("//div[@id='content']")
        category_entity = search.xpath("//h1[@id='firstHeading']/text()").extract_first()
        candidate_lists_ = search.xpath("//div[@class='mw-category-generated']//a/@href").extract()
        candidate_lists = []
        # 百科页面有许多超链接是锚链接，需要过滤掉
        for url in candidate_lists_:
            if filter(url): # 分类请求中过滤掉一些不符合的请求（例如明显包含游戏的关键词都不要爬取）
                continue
            if '/wiki' in url and 'https://zh.wikipedia.org' not in url:
                if ':' not in url or (':' in url and 'Category:' in url):
                    candidate_lists.append('https://zh.wikipedia.org' + url)
        # self.start_urls = self.urlQueue.candidates
        cates_url, content_url = split(candidate_lists)
        self.urlQueue.add_has_viewd(this_url)
        self.urlQueue.add_candidates(content_url)
        self.urlQueue.add_candidates(cates_url)
        # print('候选请求数=', len(self.urlQueue.candidates))
        # print('已处理请求数=', len(self.urlQueue.has_viewd))
        # 处理完分类页面后，将所有可能的内容请求链接直接提交处理队列处理
        if len(self.urlQueue.candidates) == 0:
            # print(111111)
            self.crawler.engine.close_spider(self)

        for url in self.urlQueue.candidates:
            if url in self.urlQueue.has_viewd:
                continue
            if 'Category:' in url:
                # print(url)
                yield scrapy.Request(url, callback=self.parse_category, dont_filter=True)
                # pass
            else:
                yield scrapy.Request(url, callback=self.parse_content, dont_filter=True)


    def parse_content(self, response):
        '''
        处理百科页面请求
        :param response:
        :return:
        '''
        counselor_item = ContentItem()
        sel = Selector(response)
        text = response.css('html').getall()
        text = ' '.join(text).strip()
        
        image_urls = []
        script_content = response.xpath('//script[@type="application/ld+json"]/text()').get()
        # print(script_content)
        if script_content:
            # 解析JSON数据
            try:
                json_data = json.loads(script_content)

                # 提取所需字段或执行其他操作
                for i in json_data:
                    
                    if(i=="@graph"):
                        # print(type(json_data[i][1]))
                        for k in json_data[i][1]:
                            # @type,name,description,image,address,geo
                            if(k=="image"):
                                # print(json_data[i][1][k])
                                image_urls = json_data[i][1][k]
            except:
                print(script_content)
                # input()
        
        # for image_url in image_urls:
        #     image_url = self.add_scheme(image_url, response.url)
        #     yield scrapy.Request(image_url, callback=self.save_image)
        
        # print(text)
        # input()
        this_url = response.url
        self.urlQueue.delete_candidate(this_url)
        # print('this_url=', this_url)
        
        from lxml import etree
        from io import StringIO
        # 将HTML字符串解析为XML树
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(str(sel) ), parser)
        # 使用XPath表达式提取<meta>标签内容
        meta_tags = sel.xpath('//meta')
        search = ""
        # 遍历提取的<meta>标签内容
        for meta_tag in meta_tags:
            name = meta_tag.attrib.get('name')
            content = meta_tag.attrib.get('content')
            if(type(name)==str):
                search = search + name + ":"  
            if(type(content)==str):
                search = search + content  + "\n"
        search = search + extract_main_content(str(sel))
        content_entity = sel.xpath("//title").extract_first().strip('<title>').strip('</title>').strip(' ').strip('\n').strip(' ').strip('	')
        content_entity = extract_chinese_from_html(content_entity)

        if(content_entity):
            content_entity = content_entity[0]
        else:
            content_entity = ""
        # content_entity = search.xpath("//h1[@id='firstHeading']/span/text()").extract_first()


        # content_page = Traditional2Simplified(search)

        # content_page = content_entity # search.xpath("//div[@id='bodyContent']//div[@class='mw-body-content']//div[@class='mw-content-ltr mw-parser-output']").extract_first()# 将带有html的标签的整个数据拿下，后期做处理

        # cates = search.xpath("//div[@id='catlinks']//ul//a/text()").extract()
        cates = []
        # candidate_lists_ = search.xpath("//div[@id='bodyContent']//*[@id='mw-content-text' and not(@class='references') and not(@role='presentation')]//a/@href").extract()
        # candidate_lists = []
        # 百科页面有许多超链接是锚链接，需要过滤掉
        # for url in candidate_lists_:
        #     if '/wiki' in url and 'https://zh.wikipedia.org' not in url:
        #         if ':' not in url or (':' in url and 'Category:' in url):
        #             candidate_lists.append('https://zh.wikipedia.org' + url)

        # self.start_urls = self.urlQueue.candidates
        self.urlQueue.add_has_viewd(this_url)
        # self.urlQueue.add_candidates(candidate_lists)
        # print('候选请求数=', len(self.urlQueue.candidates))
        # print('已处理请求数=', len(self.urlQueue.has_viewd))
        self.urlQueue.save_has_viewd()
        # 将当前页面的信息保存下来
        # print(content_entity)
        # 如果当前的content的标题或分类属于需要过滤的词（例如我们不想爬取跟游戏有关的，所以包含游戏的请求或分类都不保存）
        is_url_filter = filter(content_entity)
        is_cates_filter = False
        for cate in cates:
            cate = Traditional2Simplified(cate)
            if filter(cate):
                is_cates_filter = True
                break
        if is_url_filter == False and is_cates_filter == False:
            
            counselor_item['content_entity'] = content_entity.replace(':Category', '')
            counselor_item['category'] = '\t'.join(cates)

            counselor_item['time'] = str(time.time())
            counselor_item['url'] = this_url
            counselor_item['content'] = Traditional2Simplified(str(search))
            # return counselor_item

        # print(type(text))
        soup = BeautifulSoup(text, 'html.parser')
        sel = soup.get_text(separator=' ')
        sel = " ".join(sel.split())
        # print(sel)
        from datetime import date

        current_date = date.today()
        # print(current_date)
        origin_dir = '/nfs-data/spiderman/origin_page/'+str(current_date)+'/'+str(self.id)+"/"+self.building_name+"/"
        dir = '/nfs-data/spiderman/content/'+str(current_date)+'/'+str(self.id)+"/"+self.building_name+"/"
        import os
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(origin_dir):
            os.makedirs(origin_dir)
        # print(counselor_it'content_entity'])
        # print(counselor_item['content_entity'])
        # print(counselor_item['url'])
        # print(Traditional2Simplified(counselor_item['content_entity']))
        # input()
        # print(counselor_item['url'],Traditional2Simplified(self.building_name) ,Traditional2Simplified(counselor_item['content_entity']),(False==("wikipedia" in counselor_item['url'])),((Traditional2Simplified(self.building_name) in  Traditional2Simplified(counselor_item['content_entity']))))
        if  ("https://buy.housefun.com.tw/NoSupport/NoBuild.aspx"!=counselor_item['url'] and (False==("wikipedia" in counselor_item['url']) or (Traditional2Simplified(self.building_name) in  Traditional2Simplified(counselor_item['content_entity'])))):
            
            with open(dir + counselor_item['content_entity'] + '(' + counselor_item['time'] + ')' + '.txt', 'w', encoding='utf-8') as fw:
                # print(counselor_item)
                fw.write('建筑名称: '+self.building_name+"\n")
                if "wikipedia" in counselor_item['url']:
                    fw.write('来源: wikipedia'+"\n")
                elif "housefun" in counselor_item['url']:
                    fw.write('来源: 好房网'+self.building_name+"\n")
                elif "houseprice" in counselor_item['url']:
                    fw.write('来源: 比价王'+self.building_name+"\n")
                fw.write('标题：\n' + counselor_item['content_entity'] + '\n')
                fw.write('原文地址：' + counselor_item['url'] + '\n')
                fw.write('爬取时间：' + counselor_item['time'] + '\n\n')
                fw.write(counselor_item['content'])

            # 使用示例
            source_folder = '/nfs-data/spiderman/content/'+str(current_date)+'/'+str(self.id)+"/"
            destination_folder = '/nfs-data/spiderman/content/temp/'+str(self.id)+"/"  # 替换为目标文件夹路径
            
            copy_folder(source_folder, destination_folder)

            with open(origin_dir + counselor_item['content_entity'] + '(' + counselor_item['time'] + ')' + '.txt', 'w', encoding='utf-8') as fw:
                fw.write('标题：\n' + counselor_item['content_entity'] + '\n')
                fw.write('原文地址：' + counselor_item['url'] + '\n')
                fw.write('爬取时间：' + counselor_item['time'] + '\n\n')
                fw.write(str((text)))
            
        # 处理完分类页面后，将所有可能的内容请求链接直接提交处理队列处理
        for url in self.urlQueue.candidates:
            # print(url)
            if 'Category:' in url:
                # print(url)
                yield scrapy.Request(url, callback=self.parse_category, dont_filter=True)
            else:
                yield scrapy.Request(url, callback=self.parse_content, dont_filter=True)
