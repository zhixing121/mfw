# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from pymongo import MongoClient
from elasticsearch import Elasticsearch
import pyelasticsearch

client = MongoClient()
# mfw_db = client['mfw0728_sc_new']
mfw_db = client['mfw0728_sc']

es = Elasticsearch(['192.1.211.151'],http_auth=('elastic', 'password'),port=9200)

class MafengwoPipeline(object):
    def process_item(self, item, spider):
        if item.__class__.__name__ == "MafengwoItem":
            mfw_db[item.__class__.__name__].update({'poi_id': item['poi_id']}, dict(item), upsert=True)
            print(item['scene_spot_name'])

        if item.__class__.__name__ == "CommentItem":
            # print(item)
            # mfw_db[item.__class__.__name__].insert(dict(item))
            mfw_db[item.__class__.__name__].update({'comment_text': item['comment_text']}, dict(item), upsert=True)

        if item.__class__.__name__ == "HotelAreaItem":
            # print(item)
            mfw_db[item.__class__.__name__].update({'mddid': item['mddid']}, dict(item), upsert=True)
        if item.__class__.__name__ == "CustomItem":
            # print(item)
            mfw_db[item.__class__.__name__].update({'custom_id': item['custom_id']}, dict(item), upsert=True)

        return item


class MfwEsPipline(object):
    def process_item(self,item,spider):
        if item.__class__.__name__ == "CommentItem":
            es.index(index='indexName', doc_type='typeName', body=dict(item))
        return item

