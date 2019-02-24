from pymongo import MongoClient

CONN = MongoClient('127.0.0.1', 27017)
DB = CONN.mfw0725


class DealAreaPathsToTxt(object):
    def __init__(self):
        self.data_list = DB['scene_area'].find({})

    def deal_area_paths(self, area_paths):
        line_str = ''
        for index, lat_lng in enumerate(area_paths):
            line_str = line_str + str(index) + ' ' + str(lat_lng['lng']) + ' ' + str(lat_lng['lat']) + ' 1.#QNAN 1.#QNAN' + '\n'
        return line_str

    def run(self):
        with open(r'F:\mfw\scene_area_new.txt','w+',encoding='utf-8') as f:
            f.write('Polygon\n')
            for area_index, data in enumerate(self.data_list):
                f.write(str(data['area_id'])+' '+'0\n')
                line_str = self.deal_area_paths(data["area_paths"])
                f.write(line_str)
            f.write('END')
        f.close()

if __name__ == '__main__':
    daptt = DealAreaPathsToTxt()
    daptt.run()
