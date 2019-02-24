# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MafengwoItem(scrapy.Item):
    # define the fields for your item here like:

    scene_spot_name = scrapy.Field()
    # scene_spot_url = scrapy.Field()
    space_time = scrapy.Field()
    price = scrapy.Field()
    traffic = scrapy.Field()
    open_time = scrapy.Field()
    position = scrapy.Field()
    within = scrapy.Field()
    # comment = scrapy.Field()
    comment_num_count = scrapy.Field()
    poi_id = scrapy.Field()
    mddid = scrapy.Field()
    lat = scrapy.Field()
    lng = scrapy.Field()



class CommentItem(scrapy.Item):
    poi_id = scrapy.Field()
    comment_text = scrapy.Field()
    comment_time = scrapy.Field()
    comment_star = scrapy.Field()
    custom_id = scrapy.Field()


class HotelAreaItem(scrapy.Item):
    mddid = scrapy.Field()
    price = scrapy.Field()
    paths = scrapy.Field()
    lat = scrapy.Field()
    lng = scrapy.Field()
    name = scrapy.Field()


class CustomItem(scrapy.Item):
    custom_id = scrapy.Field()
    custom_cty = scrapy.Field()
    custom_sex = scrapy.Field()

