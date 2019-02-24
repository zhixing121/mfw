import jieba
import numpy as np
import pandas as pd
from pymongo import MongoClient
import psycopg2
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

CITY_PATH = r"F:\mfw\city_utf8.txt"
CITY_JSON_PATH = r"F:\mfw\city_xian.json"
CONN = psycopg2.connect(host='127.0.0.1', port='5432', user='postgres', password='123', database='mfw')
CUR = CONN.cursor()


class MfwNpPd(object):
    def __init__(self):
        pass

        # 获取景点对应的评论以及客户信息

    @staticmethod
    def get_scene_comment_custom_data():
        CUR.execute(
            """select * from 
               (select t1.scene_spot_name,t1.poi_id,t2.comment_text,t2.comment_time,t2.comment_star,t2.custom_id,t3.custom_cty,t3.custom_sex
                              FROM mafengwoitem as t1 
                              JOIN 
                              commentitem as t2 
                              ON(t1.poi_id=t2.poi_id)
                              JOIN
                              custom_info_new as t3
                              ON(t2.custom_id=t3.custom_id)
               )
               as a1
               WHERE a1.poi_id IN
               (select poi_id FROM(select count(poi_id),poi_id FROM commentitem GROUP BY poi_id) as c1 WHERE c1.count>10 ORDER BY c1.count DESC)"""
        )
        return CUR.fetchall()

    # 处理获取的txt地名信息，处理为txt
    @staticmethod
    def read_city_to_dict():
        with open(CITY_PATH, 'r', encoding='utf-8') as f:
            text_list = []
            text_str = f.read().split('/')
            for k, v in zip(text_str[1::2], text_str[2::2]):
                province_dict = {}
                city = v.split('\n')
                # xian_list = []
                xian_dict = {}
                for xian in city:
                    # city_dict = {}
                    xian_str = xian.split("\n")
                    # print(xian_str)
                    if xian_str[0] != '':
                        city_xian = xian_str[0].split(':')
                        if city_xian[0] != ' ':
                            # city_dict[city_xian[0]] = city_xian[1].strip().split(' ')
                            xian_dict[city_xian[0]] = city_xian[1].strip().split(' ')
                        # xian_list.append(city_dict)
                province_dict[k] = xian_dict
                text_list.append(province_dict)
            f.close()
        return text_list

    def write_dict_to_jsonfile(self, dict_data, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(dict_data, ensure_ascii=False))
            f.close()

    # 从数据库获取用户城市信息
    @staticmethod
    def get_city():
        CUR.execute("""
        select custom_id,custom_cty from custom_info_new
        """)
        return CUR.fetchall()


    def update_province_to_custom_table(self,province_name,custom_id):
        """
        更新省级数据到游客表
        :param province_name:所属省名称
        :param custom_id:游客id
        :return:
        """
        CUR.execute(
            """
            UPDATE custom_info_new SET custom_province= %s WHERE "custom_id"= %s
            """,(province_name,custom_id)
        )
        CONN.commit()

    # 分别获取省市县列表
    @staticmethod
    def get_province_city_county_list():
        with open(CITY_JSON_PATH, 'r', encoding='utf-8') as fff:
            city_dict = json.load(fff)
            province_list = [list(k.keys())[0] for k in city_dict]
            city_next_list = []
            county_list = []
            for i in [list(k.values())[0] for k in city_dict]:
                for x in i:
                    city_next_list.append(list(x.keys())[0])
                    for y in list(x.values())[0]:
                        county_list.append(y)
            return (county_list, province_list, city_next_list)

    def deal_words(self, city_words, county_list, province_list, city_next_list):  # 县，省，市
        """
        匹配结果
        :param city_words: 输入的省市词
        :param county_list:县级列表
        :param province_list:省级
        :param city_next_list:市级列表
        :return:返回匹配出的结果【‘城市’，分数】
        """
        if city_words == None:
            return None
        if isinstance(city_words, str):
            result = self.deal_process_city_words(city_words, county_list, province_list, city_next_list)
            if result == [('北京', 0)]:
                return None
            return result
        if isinstance(city_words, list):
            for city in city_words:
                if city == '' or city == ' ' or city == '-':
                    continue
                else:
                    result = self.deal_process_city_words(city, county_list, province_list, city_next_list)
                    if result == [('北京', 0)]:
                        return None
                    return result

    def process_words(self, words, choice_list, limit=1):
        result = process.extract(words, choice_list, limit=limit)
        return result

    def deal_process_city_words(self, city_words, province_list, city_next_list, county_list):
        """

        :param city_words:
        :param province_list:
        :param city_next_list:
        :param county_list:
        :return:
        """
        province_result = self.process_words(city_words, province_list, 1)
        city_result = self.process_words(city_words, city_next_list, 1)
        county_result = self.process_words(city_words, county_list, 1)
        belong_city = self.get_city_result(province_result, city_result, county_result)
        return belong_city

    def get_city_result(self, province_result, city_result, county_result):
        """
        根据匹配的分数选择结果：分数为100或者大于60的三级中最匹配的一级,否则返回None
        :param province_result:省级结果
        :param city_result:市级结果
        :param county_result:县级结果
        :return:返回最匹配的结果
        """
        if province_result[0][1] == 100 or (province_result[0][1] == max(province_result[0][1], city_result[0][1],
                                                                        county_result[0][1]) and province_result[0][1]>60):
            return province_result
        if city_result[0][1] == 100 or (city_result[0][1] == max(province_result[0][1], city_result[0][1],
                                                          county_result[0][1]) and city_result[0][1]>60):
            return city_result
        if county_result[0][1] == 100 or (county_result[0][1] == max(province_result[0][1], city_result[0][1],
                                                              county_result[0][1]) and county_result[0][1]>60):
            return county_result
        if province_result[0][1] <=60:
            if city_result[0][1] <=60 or county_result[0][1] <=60:
                return None

    def get_province_from_dict(self,city,city_dict):
        """
        输入得到的result—城市，输出所属省份
        :param city:所属城市
        :return:所属省份
        """
        for province_items in city_dict:
            for province_key,province_val in province_items.items():
                if city != province_key:
                    for city_county_dict in province_val:
                        for city_key,city_val in city_county_dict.items():
                            if city != city_key:
                                if city not in city_val:
                                    continue
                                else:
                                    return province_key
                            else:
                                return province_key
                else:
                    return province_key





if __name__ == '__main__':
    data_class = MfwNpPd()
    data = data_class.get_scene_comment_custom_data()
    df = pd.DataFrame(data,
                      columns=["scene_spot_name", "poi_id", "comment_text", "comment_time", "comment_star", "custom_id",
                               "custom_cty", "custom_sex"])
    df['comment_star'] = df['comment_star'].astype(int)
    group_scene = df.groupby('scene_spot_name')
    group_scene_cty = df.groupby('custom_cty')
    df['province'] = None
    # print(group_scene_cty.size())
    # print(df['custom_cty'],df["province"])
    # jieba.load_userdict(CITY_PATH)
    # city_dict = json.loads(CITY_JSON_PATH,encoding='utf-8')


    county_list, province_list, city_next_list = data_class.get_province_city_county_list()
    city_list = data_class.get_city()
    with open(CITY_JSON_PATH, 'r', encoding='utf-8') as fff:
        city_dict = json.load(fff)
        for index, city in enumerate(city_list):
            print(index, "-" * 10, city)
            if city[1] != None:  # [游客id，游客归属地]
                if len(city[1]) > 3:  # 游客归属地字符长度大于3进行分词
                    # print(city[0])
                    city_words = [x for x in jieba.cut(city[1], cut_all=False)]
                    result = data_class.deal_words(city_words, province_list, city_next_list, county_list)
                    # print(result)
                    if result != None:
                        province_name = data_class.get_province_from_dict(result[0][0],city_dict)
                        data_class.update_province_to_custom_table(province_name,city[0]) #更新到数据库中
                        print(province_name)
                    else:
                        print('不存在！！！')
                else:
                    city_words = city[1]
                    result = data_class.deal_words(city_words, province_list, city_next_list, county_list)
                    # print(result)
                    if result != None:
                        province_name = data_class.get_province_from_dict(result[0][0], city_dict)
                        data_class.update_province_to_custom_table(province_name, city[0])
                        print(province_name)
                    else:
                        print('不存在！！！')
    fff.close()


    # city = ('52409679', '深圳')
    # data_class.update_province_to_custom_table(*city)
    #
    # text_dict = data_class.read_city_to_dict()
    # data_class.write_dict_to_jsonfile(text_dict,r'F:\mfw\xian1101.json')