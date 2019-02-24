from lxml import etree
import re
import random
import json
import requests
from queue import Queue
from threading import Thread
from pymongo import MongoClient


CONN = MongoClient('127.0.0.1',27017)
DB = CONN.mfw0725
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36 OPR/52.0.2871.64x"]




class MfwSceneArea():


    def __init__(self):
        self.temp_url = "http://www.mafengwo.cn/u/{}.html"
        user_agent = random.choice(USER_AGENT_LIST)
        self.headers = {
            "User-Agent": user_agent}
        self.custom_obj = DB['CommentItem'].distinct('custom_id')
        self.exits_custom_obj = DB['custom_info1'].distinct('custom_id')


    # def get_url_list(self):
    #     for url_code in self.custom_obj:
    #         # print(url_code)
    #         if url_code not in self.exits_custom_obj:
    #             custom_url = self.temp_url.format(int(url_code))
    #             self.url_queue.put(custom_url)


    def response_url(self,url):
        #解析连接
        response = self.get_response(url)
        print(response)
        if response != None:
            print(response.status_code)
        return response



    def extract_content(self,custom_id,response):
        res = etree.HTML(response.text)
        custom_item = {}
        # custom_item['custom_id'] = response.request.url.split('/')[4].replace('.html', '')
        custom_item['custom_id'] = custom_id
        try:
            custom_item['custom_cty'] = res.xpath("//span[@class='MAvaPlace flt1']/@title")[0]
        except Exception as e:
            # print(e)
            custom_item['custom_cty'] = ''
        try:
            custom_item['custom_sex'] = res.xpath("//div[@class='MAvaName']/i/@class")[0].replace(
                'MGender', '')
        except Exception as e:
            # print(e)
            custom_item['custom_sex'] = ''
        DB["custom_info1"].update({'custom_id': custom_item['custom_id']}, custom_item, upsert=True)
        # DB["custom_info1"].insert(custom_item)


    # 获取代理列表
    @staticmethod
    def get_proxy():
        return requests.get("http://127.0.0.1:5010/get/").text

    # 删除代理
    @staticmethod
    def delete_proxy(proxy):
        requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))

    # 用代理解析网页，并获取response
    def get_response(self,url):
        retry_count = 5
        proxy = self.get_proxy()
        print(proxy)
        while retry_count > 0:
            try:
                response = requests.get(url, headers=self.headers, proxies={"http":"http://{}".format(proxy)},timeout=3)
                return response
            except Exception:
                retry_count -= 1
        self.delete_proxy(proxy)
        return None

    def run(self):
        for url_code in self.custom_obj:
            # print(url_code)
            if url_code not in self.exits_custom_obj:
                custom_url = self.temp_url.format(int(url_code))
                print(custom_url)
                response = self.response_url(custom_url)
                if response != None:
                    self.extract_content(url_code,response)


if __name__ == '__main__':
    mfw_scene_area = MfwSceneArea()
    mfw_scene_area.run()