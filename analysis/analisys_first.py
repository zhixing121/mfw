import matplotlib

import psycopg2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from scipy.misc import imread
from PIL import Image

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'


from sklearn.feature_extraction.text import CountVectorizer

"""
各个省区的统计：总人数，男女，不同年份
各个省区分数，各个景区分数等

文本分析，情感分析

词云

"""


CONN = psycopg2.connect(host='127.0.0.1', port='5432', user='postgres', password='123', database='mfw')
CUR = CONN.cursor()
STOP_WORDS = r'F:\mfw\stop_words\stop_words.txt'
GL_IMG = r'F:\mfw\xbs.jpg'
ZMPJ = r'F:\mfw\sentiment1\zmpj.txt'
SCENE_NAME = r'F:\mfw\scene_spot_name.txt'
# jieba.load_userdict(ZMPJ)
# jieba.load_userdict(SCENE_NAME)


class AnalisysMfwData(object):
    def __init__(self):
        pass

    @staticmethod
    def get_scene_and_comment_data():
        """
        获取景点数据以及评论数据
        :return:
        """
        CUR.execute(
            """select * from 
               (select t1.scene_spot_name,t1.poi_id,t2.comment_text,t2.comment_time,t2.comment_star
               FROM mafengwoitem as t1 
               JOIN 
               commentitem as t2 
               ON(t1.poi_id=t2.poi_id))
               as a1
               WHERE a1.poi_id IN
               (select poi_id FROM(select count(poi_id),poi_id FROM commentitem GROUP BY poi_id) as c1 WHERE c1.count>10 ORDER BY c1.count DESC)
            """)
        return CUR.fetchall()

    @staticmethod
    def get_scene_comment_custom_data():
        """
        获取景点数据，评论数据，游客数据
        :return:
        """
        CUR.execute(
            """select * from 
               (select t1.scene_spot_name,t1.poi_id,t2.comment_text,t2.comment_time,t2.comment_star,t2.custom_id,t3.custom_cty,t3.custom_sex,t3.custom_province
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

    def word_cloud_comment(self, text, stop_key):
        """
        评论词云
        :param text:输入的文本
        :param stop_key:停用词
        :return:返回词云
        """
        wordlist_after_jieba = jieba.cut(text, cut_all=False)
        word_list = []
        # mask_img = np.array(Image.open(GL_IMG))
        number = [str(i) for i in range(10000)]
        stop_key.extend(number)
        for word in wordlist_after_jieba:
            if word not in stop_key:
                word_list.append(word)
        wl_space_split = " ".join(word_list)
        # tf_idf 统计
        print(jieba.analyse.extract_tags(wl_space_split, topK=100, withWeight=True))
        my_wordcloud = WordCloud(background_color="white",
                                 # mask=mask_img,
                                 width=1000,
                                 height=1000,
                                 font_path="C:\\Windows\\Fonts\\STFANGSO.ttf",  # 不加这一句显示口字形乱码
                                 margin=2).generate(wl_space_split)

        plt.imshow(my_wordcloud)
        plt.axis("off")
        plt.show()

    def read_stop_words(self):
        stop_words = [line.strip() for line in open(STOP_WORDS, 'r').readlines()]
        return stop_words

    @staticmethod
    def read_mfw_item_name_to_txt():
        """
        从数据库获取景点名称，并分词，用来从文本中去除，当做停用词
        :return:
        """
        CUR.execute("""select mafengwoitem.scene_spot_name FROM mafengwoitem""")
        # 保存景点名称到txt
        with open(r'F:\mfw\scene_spot_name.txt', 'w') as f:
            for name in CUR.fetchall():
                if name is not None or name != '':
                    f.write(name[0])
                    f.write('\n')
        f.close()
        print("景点名称写完")

    def run(self):
        scene_and_comment_data = self.get_scene_and_comment_data()
        data_list = []
        poi_id_set = set(i[1] for i in scene_and_comment_data)
        for poi_id in poi_id_set:
            data_dict = {}
            data_dict['poi_id'] = poi_id
            scene_spot_content_list = []
            for v in (scene_and_comment_data):
                data_dict['scene_spot_name'] = v[0]
                if v[1] == poi_id:
                    scene_spot_content_list.append(v[2])
                data_dict['scene_spot_content'] = scene_spot_content_list
            data_list.append(data_dict)
        return data_list


if __name__ == '__main__':
    amdata = AnalisysMfwData()
    # data = amdata.run()
    # print(len(data))
    # amdata.read_mfw_item_name_to_txt()
    # text_str = ''
    # for d in data:
        # print(d['scene_spot_content'])
        # text_str += '.'.join(str(s) for s in d['scene_spot_content'] if s is not None)
    # print(len(text_str))
    # text_comment = '。'.join(text_list)
    # stopwords = amdata.read_stop_words()
    # amdata.word_cloud_comment(text_str, stopwords)
    # amdata.read_stop_words()


    get_scene_comment_custom_data = amdata.get_scene_comment_custom_data()
    # print(type(get_scene_comment_custom_data))
    # print(get_scene_comment_custom_data)
    df_mfw = pd.DataFrame(get_scene_comment_custom_data, columns=["scene_spot_name", "poi_id", "comment_text", "comment_time", "comment_star", "custom_id", "custom_cty", "custom_sex", "custom_province"])
    # df_mfw = pd.DataFrame(get_scene_comment_custom_data, columns=["scene_spot_name", "poi_id", "comment_text", "comment_time", "comment_star", "custom_id", "custom_cty", "custom_sex", "custom_province"])
    df_mfw['comment_star'] = df_mfw['comment_star'].astype(int)
    # pd.Index(df_mfw,df_mfw['poi_id'])
    df_mfw['year'],df_mfw['another_date'] = df_mfw['comment_time'].str.split('-',n=1).str # 切割时间
    df_mfw1 = df_mfw.set_index(['poi_id','custom_id'])  # 设置列为索引并生成新的df
    # print(df_mfw1.dropna(how='all').reset_index())
    # melted = pd.melt(df_mfw,['poi_id'])
    # print(df_mfw)
    province_count = df_mfw['custom_province'].value_counts()
    sex_count = df_mfw['custom_sex'].value_counts()
    time_count = df_mfw['comment_time'].str.split('-', expand=True)[0].value_counts(sort=False)
    # print(time_count)
    # province_count.plot(kind='bar').get_figure().savefig(r'E:\python_hexin\mafengwo\sysy.png')
    # sex_count.plot(kind='bar').get_figure().savefig(r'E:\python_hexin\mafengwo\sexsy.png')
    # time_count.plot(kind='bar').get_figure().savefig(r'E:\python_hexin\mafengwo\timesy.png')


    group1 = df_mfw.groupby('scene_spot_name')  # 按照景点分组
    # print([x for x in group1])
    # print(group1['comment_star'].agg('mean'))
    # print([x for x in group1['custom_cty']])
    # group2 =
    group_star = df_mfw.groupby(['poi_id'])['comment_star'].agg('mean')
    # print(group_star.plot(kind='hist').get_figure().savefig(r'E:\python_hexin\mafengwo\ssss.png'))
    # print(df_mfw.groupby(['custom_province','custom_sex'])['custom_sex'].count().unstack()) # 每个省的男女
    # print(df_mfw.groupby(['year','custom_province','custom_sex'])['custom_sex'].count().unstack()) # 每年不同省份的人
    print(df_mfw.pivot_table(['comment_star'],index=['year','custom_sex'],columns='custom_province',aggfunc='count',fill_value=0))
