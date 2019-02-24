# -*- coding: utf-8 -*-
import json

import re

import copy

import math
from bs4 import BeautifulSoup
import scrapy
from mafengwo.items import MafengwoItem, CommentItem, HotelAreaItem, CustomItem


class MfwSpider(scrapy.Spider):
    name = 'mfw'
    allowed_domains = ['mafengwo.cn']

    # start_urls = ['http://www.mafengwo.cn/ajax/router.php']

    def start_requests(self):  # 重写
        url = "http://www.mafengwo.cn/ajax/router.php"
        post_data = {
            "sAct": "KMdd_StructWebAjax|GetPoisByTag",
            "iMddid": "12810",
            "iTagId": "0",
            "iPage": "1"
        }
        yield scrapy.FormRequest(
            url=url,
            formdata=post_data,
            callback=self.parse_all_scene
        )

    def parse_all_scene(self, response):
        # 爬取本页所有景点的链接
        scene_spot_li = json.loads(response.body.decode())['data']['list']
        scene_spot_page = json.loads(response.body.decode())['data']['page']
        next_page = re.findall("<a class=\"pi pg-next\" data-page=\"(.*)\" rel", scene_spot_page)
        scene_spot_url_list = re.findall("<a href=\"(.*)html", scene_spot_li)
        # scene_spot_name_list = re.findall("title=\"(.*)\"",scene_spot_li)
        for spot_url in scene_spot_url_list:
            # item = MafengwoItem()
            scene_spot_url = "http://www.mafengwo.cn" + spot_url + "html"
            yield scrapy.Request(  # 爬取详情页
                url=scene_spot_url,
                # meta={'item': item},
                callback=self.parse_detail
            )
        # 下一页景点
        if next_page != []:
            post_data = {
                "sAct": "KMdd_StructWebAjax|GetPoisByTag",
                "iMddid": "12810",
                "iTagId": "0",
                "iPage": next_page[0]
            }
            next_url = "http://www.mafengwo.cn/ajax/router.php"
            yield scrapy.FormRequest(
                url=next_url,
                formdata=post_data,
                callback=self.parse_all_scene
            )


            # 爬取详情页

    def parse_detail(self, response):
        item = MafengwoItem()
        # item = copy.deepcopy(response.meta['item'])
        html = response.body.decode()
        item["scene_spot_name"] = response.xpath("//div[@class='title']/h1/text()").extract_first()
        # item["space_time"] = ''
        try:
            item["price"] = response.xpath("//div[@class='mod mod-detail']/dl")[1].xpath(
                "./dd/div/text()").extract_first()
        except Exception as e:
            # print(e)
            item["price"] = ''
        try:
            item["traffic"] = response.xpath("//div[@class='mod mod-detail']/dl")[0].xpath(
                "./dd/text()").extract()
        except Exception as e:
            # print(e)
            item["traffic"] = ''
        try:
            item["open_time"] = response.xpath("//div[@class='mod mod-detail']/dl")[2].xpath(
                "./dd/text()").extract_first()
        except Exception as e:
            # print(e)
            item["open_time"] = ''

        item["position"] = response.xpath("//div[@class='mod mod-location']/div/p/text()").extract_first()
        try:
            comment_num_count = \
                re.findall('\d+', response.xpath("//div[@id='poi-navbar']//span/text()").extract_first())[0]
        except Exception as e:
            # print(e)
            comment_num_count = ''
        mddid = json.loads(re.findall("window.Env = (.*);", response.xpath("//script/text()").extract_first())[0])[
            'mddid']
        poiid = json.loads(re.findall("window.Env = (.*);", response.xpath("//script/text()").extract_first())[0])[
            'poiid']
        item['poi_id'] = poiid
        item['mddid'] = mddid
        item['comment_num_count'] = comment_num_count
        # 周围景区area网址
        around_area_url = "http://www.mafengwo.cn/hotel/ajax.php?sAction=getGMapArea&iMddId={}".format(int(mddid))
        yield scrapy.Request(
            url=around_area_url,
            callback=self.parse_around_area
        )

        if int(comment_num_count) > 0 or comment_num_count is not None:  # 评论总数不为空或者大于0
            comment_url = "http://pagelet.mafengwo.cn/poi/pagelet/poiCommentListApi"
            first_comment_params = comment_url + '?params={' + '"poi_id":"{}","page":1'.format(
                poiid) + ',"just_comment":1}'
            yield scrapy.Request(
                url=first_comment_params,
                callback=self.parse_comment
            )

        yield item
        # print(item)

    def parse_around_area(self, response):
        content = json.loads(response.body.decode())
        if content != []:
            for k, v in content.items():
                if k != '0':
                    area_item = HotelAreaItem()
                    area_item['name'] = v['name']
                    area_item['mddid'] = v['mddid']
                    area_item['price'] = v['price']
                    area_item['paths'] = v['paths']
                    area_item['lat'] = v['lat']
                    area_item['lng'] = v['lng']
                    yield area_item

    def parse_comment(self, response):
        poi_id = json.loads(response.body.decode())['data']['controller_data']['poi_id']
        comment_count = json.loads(response.body.decode())['data']['controller_data']['comment_count']
        soup = BeautifulSoup(json.loads(response.body.decode())['data']['html'], 'lxml')
        comment_li = soup.find_all('li', {"class", 'rev-item comment-item clearfix'})
        for li in comment_li:
            comment_item = CommentItem()
            comment_text = re.sub(r'\xa0|\s', ',', li.find('p', {"class", 'rev-txt'}).get_text())
            comment_time = li.find('span', {"class", 'time'}).get_text()
            comment_star = re.findall('\d', str(li.find('span', {'class', "s-star"})))[0]
            custom_href = li.find('a', {'class', 'name'}).get('href')
            custom_id = re.findall('\d+', custom_href)[0]
            comment_item["poi_id"] = poi_id
            comment_item["comment_text"] = comment_text
            comment_item["comment_time"] = comment_time
            comment_item["comment_star"] = comment_star
            comment_item["custom_id"] = custom_id

            # 爬取用户信息
            yield scrapy.Request(
                url="http://www.mafengwo.cn" + custom_href,
                callback=self.parse_custom
            )

            yield comment_item
            # print(comment_item)

        for page_num in range(1, math.ceil(int(comment_count) / 15) + 1):
            comment_url = "http://pagelet.mafengwo.cn/poi/pagelet/poiCommentListApi"
            first_comment_params = comment_url + '?params={' + '"poi_id":"{}","page":{}'.format(
                poi_id, page_num) + ',"just_comment":1}'
            yield scrapy.Request(
                url=first_comment_params,
                callback=self.parse_comment
            )

    def parse_custom(self, response):
        custom_item = CustomItem()
        custom_item['custom_id'] = response.request.url.split('/')[4].replace('.html', '')
        try:
            custom_item['custom_cty'] = response.xpath("//span[@class='MAvaPlace flt1']/@title").extract_first()
        except Exception as e:
            # print(e)
            custom_item['custom_cty'] = ''
        try:
            custom_item['custom_sex'] = response.xpath("//div[@class='MAvaName']/i/@class").extract_first().replace(
                'MGender', '')
        except Exception as e:
            # print(e)
            custom_item['custom_sex'] = ''
        yield custom_item
