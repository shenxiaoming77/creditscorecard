import numpy as np
import re
import time
import datetime
import pandas as pd
import decimal
from dateutil.relativedelta import relativedelta

city_0=['北京','上海','广州','深圳']

city_01=['成都市','杭州市','重庆市','武汉市','苏州市','西安市','天津市','南京市','郑州市','长沙市','沈阳市','青岛市','宁波市','东莞市','无锡市']

city_1=['江汉','济南','厦门','哈尔滨','长春','福州','石家庄','佛山','烟台','合肥','昆明','唐山','济南'
       ,'大连','厦门','哈尔滨','长春','福州','石家庄'
       ,'佛山','烟台','太原','合肥','南昌','南宁','昆明','温州','淄博','唐山']

city_2=[ '省直','延吉','涪陵','珲春','舟山','淮阴', '阜新','襄樊','河源','梧州','三亚','延边','衡阳','梅州','乌鲁木齐','贵阳','海口','兰州','银川','西宁','呼和浩特','泉州','包头','南通','大庆','徐州','潍坊','常州','鄂尔多斯'
    ,'绍兴','济宁','盐城','邯郸','临沂','洛阳','东营','扬州','台州','嘉兴','沧州','榆林','泰州','镇江','昆山','江阴','张家港'
    ,'义乌','金华','保定','吉林','鞍山','泰安','宜昌','襄阳','中山','惠州','南阳','威海','德州','岳阳','聊城','常德','漳州'
    ,'滨州','茂名','淮安','江门','芜湖','湛江','廊坊','菏泽','柳州','宝鸡','珠海','绵阳']

city_3=['德令哈','仙桃','康定','伊犁', '中卫', '黔江','万州','思茅','迪庆', '新余','怒江','临沧','双鸭山','甘孜','铜仁','商洛','景德镇','漯河','潮州','白银','湘西', '西双版纳','葫芦岛','抚州','黄山','白山','淮南', '恩施','眉山','平凉','河池','钦州','巢湖', '宣城','亳州','海北','株洲','枣庄','许昌','通辽','湖州','新乡','咸阳','松原','连云港','安阳','周口','焦作','赤峰','邢台','郴州','宿迁'
    ,'赣州','平顶山','桂林','肇庆','曲靖','九江','商丘','汕头','信阳','驻马店','营口','揭阳','龙岩','安庆','日照','遵义'
    ,'三明','呼伦贝尔','长治','湘潭','德阳','南充','乐山','达州','盘锦','延安','上饶','锦州','宜春','宜宾','张家口','马鞍山'
    ,'吕梁','抚顺','临汾','渭南','开封','莆田','荆州','黄冈','四平','承德','齐齐哈尔','三门峡','秦皇岛','本溪','玉林','孝感'
    ,'牡丹江','荆门','宁德','运城','绥化','永州','怀化','黄石','泸州','清远','邵阳','衡水','益阳','丹东','铁岭','晋城','朔州'
    ,'吉安','娄底','玉溪','辽阳','南平','濮阳','晋中','资阳','都江堰','攀枝花','衢州','内江','滁州','阜阳','十堰','大同'
    ,'朝阳','六安','宿州','通化','蚌埠','韶关','丽水','自贡','阳江','毕节']

city_4=['奎屯','海晏', '梅河', '锡', '林芝', '景洪','博尔','黄南', '格尔','济源','兴义','凯里', '贵池','都匀','吉首','汉中', '来宾市','甘南','德宏','大兴安岭','海南','玉树', '甘南'
    , '陇南','海西', '阳泉','铜陵', '金昌','凉山','七台河', '鄂州','莱芜','伊春', '铜川','临夏','海东'
    ,'崇左','辽源','忻州','汕尾','固原','安康','定西','庆阳','楚雄','池州','鸡西','贺州','防城港','萍乡'
    ,'红河','随州','咸宁','文山', '黔东','张家界','达川','安顺','白城','黔南','贵港', '阿坝','云浮', '鹤壁'
    ,'佳木斯','黔西','六盘水','黑河','鹤岗','淮北','鹰潭','拉萨','克拉玛依','库尔勒','昌吉','哈密','伊宁','喀什','阿克苏','石河子','晋江','增城','诸暨','丹阳','玉环','常熟'
    ,'崇明','余姚','奉化','海宁','浏阳市','大理','丽江','普洱','保山','昭通','西昌','雅安','广安','广元','巴中','遂宁'
    ,'天水','酒泉','嘉峪关','武威','张掖','石嘴山','吴忠','北海','百色','虎门镇','长安镇','鳌江-龙港镇']


def ConvertCity(x):  #一线城市值越小

    if x in city_0:
       return 0
    elif x in city_01:
       return 1
    elif x in city_1:
       return 1
    elif x in city_2:
       return 2
    elif x in city_3:
       return 3
    elif x in city_4:
       return 3
    elif x in " ".join(city_0):
       return 0
    elif x in " ".join(city_01):
       return 1
    elif x in " ".join(city_1):
       return 1
    elif x in " ".join(city_2):
       return 2
    elif x in " ".join(city_3):
       return 3
    else:
       return 3

def ConvertProvince(x):   #x表示违约率,x越小，该省赋予的得分越大
        if x< 10: #x<10
            return 3
        elif x<15:   #10<=x<15
            return 2
        elif x<20:  #15<=x<20
            return 1
        else:
            return 0   #>=20

def ConvertDateDiff(x):
    if x==0 :
        return 0
    elif x==1:
        return 1
    elif x==2:
        return 1
    elif x==3:
        return 2
    elif x==4:
        return 2
    elif x==5:
        return 2
    elif x==6:
        return 3
    else:
        return 4
#
# def ConvertAge(x):
#     if 18<=x<=40:
#         return 0
#     elif 40<x<50:
#         return 1
#     elif 50<=x<=60:
#         return 2

def ConvertAge(x):
    if 18<=x<20:
        return 0
    elif 20<=x<=40:
        return 1
    elif 40<x<50:
        return 2
    elif 50<=x<=55:
        return 3
    elif 55<x<=60:
        return 4

def ConvertApplyDateHour(x):
    if 0<=x<=6:
        return 0
    elif 6<x<=10:
        return 1
    elif 10<x<=20:
        return 2
    elif 20<x<=23:
        return 3

def ConvertregisterDateHour(x):
    if 0<=x<2 or 20<=x<=23:
        return 0
    elif 2<=x<=6:
        return 1
    elif 6<x<=10:
        return 2
    elif 10<x<=19:
        return 3

#百融信用分
#[300,500) 高风险，建议拒绝
#[500,550) 中风险，建议关注
#[550,1000] 低风险风险，建议通过

# def ConvertBrScore(x):
#     if x=="Unknown":
#         return 0
#     elif 300<=x<500:
#         return 0
#     elif 500<=x<550:
#         return 1
#     elif 550<=x<=1000:
#         return 2
#     else:
#         return 0

def ConvertBrScore(x):
    if x=="Unknown":
        return 0
    elif 300<=x<500:
        return 1
    elif 500<=x<600:
        return 2
    elif 600<=x<=700:
        return 3
    elif 700<=x<=1000:
        return 4

#芝麻信用分
#第一级：300分~500分
#第二级：500分~600分
#第三级：600分~700分
#第四级：700分~800分
#第五级：800分~950分
# def ConvertZhimaScore(x):
#     if x=="Unknown":
#         return 2
#     elif 300<=x<500:
#         return 0
#     elif 500<=x<600:
#         return 1
#     elif 600<=x<=700:
#         return 2
#     elif 700<=x<=800:
#         return 3
#     elif 800<=x<=950:
#         return 4
#     else:
#         return 2

def ConvertZhimaScore(x):

    if 300<=x<600:
        return 0
    elif 600<=x<650:
        return 1
    elif 650<=x<=700:
        return 2
    elif 700<=x<=950:
        return 3

def ConvertCompanyCollegeLength(x):
    if x=="Unknown":
         return 0
    elif 0<=x<16:
        return 1
    elif 16<=x<=23:
        return 2

#对于连续变量用badrate编码，其实肯定能分组分得更好，但是模型效果不一定好
def ConvertM1OverdueCount(x):
    if x>0:
        return 1
    else:
        return 0

def ConvertM3RepayremindCount(x):
    if x<=1:
        return 0
    elif 2<=x<=4:
        return 1
    elif x>4:
        return 2

def ConvertOccupationGender(x):
    if x=='Unknown_0':
        return 'Unknown'
    elif x=='Unknown_1':
        return 'Unknown'
    else:
        return x

def  areaVarMap(data,tag,var):

        provinces = data[var].unique() # 省类别
        provinces = [x for x in provinces if str(x) != 'nan']
        #print(provinces)
        neg = data[data[tag]==1]

        #贷款逾期占比
        negSer = pd.value_counts(neg[var],sort=False)

        posprovList=negSer.index

        for i in provinces:
            if not (i  in  posprovList):
                negSer[i]=0

        provinceSer =pd.value_counts(data[var],sort=False)

        pct= negSer.sort_index()/provinceSer.sort_index()*100
        # print(pct)
        pct =pct.map(lambda x: ConvertProvince(x))
        return  pct

def  badRateMap(df,tag,col):

    total = df.groupby([col])[tag].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[tag].sum()
    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup[col] = regroup.apply(lambda x: round(x.bad * 1.0 / x.total*100,1), axis=1)
    # print(regroup[col+'_'+'bad_rate'])
    return regroup[col]




def negToZero(x):
    if x<0:
        return 0
    else :
        return x

'''


Frequency count for variable network_len
(24,+)     8753
Unknown    6698
(12,24]    1332
(6,12]      410
6           350
未查得         255    #将未查得映射为UnKnown
(0,6]        14
(3,6]         9

-1   15.87804
 0   10.99196
 1   24.87805
 2   24.32432
 3   18.97635

 0   15.62927
1   19.88566
'''
mapping_dict = {
    "network_len": {
        "(24,+)": 3,
        "(12,24]": 2,
        "(6,12]": 2,
        6: 1,
        "(3,6]": 1,
        "(0,6]": 1,
        "Unknown": 4,
        "未查得": 0,

    },
    # "network_len": {
    #     "(24,+)": 4,
    #     "(12,24]": 3,
    #     "(6,12]": 2,
    #     6: 1,
    #     "(3,6]": 1,
    #     "(0,6]": 1,
    #     "Unknown": 0,
    #     "未查得": 0,
    #
    # },
    "gender":{
        "男": 1,
        "女": 0,
    },
    "job_level":{
         # '主任/主管/组长/初级管理' :'主任/主管/组长/初级管理',
         '主管' :'主任/主管/组长/初级管理',
         '总经理' :'总监/总经理/高管',
         # '总监/总经理/高管' :'总监/总经理/高管',
         '总监':'总监/总经理/高管',
         # '经理/中级管理' :'经理/中级管理',
         '经理':'经理/中级管理',
         # '法人/老板':'法人/老板',
          '实习生':'学生',
         #'学生':'学生',
         # 'Unknown':'Unknown',
         '司机' :'普通员工',
         '服务员':'普通员工',
         '厨师':'普通员工',
         '保安':'普通员工',
         '一般工人':'普通员工',
         # '普通员工':'普通员工',
         '技术工人':'普通员工',
         '行政/人力资源' :'行政/人力资源/财会人员/助理',
         '财会人员':'行政/人力资源/财会人员/助理',
         '助理':'行政/人力资源/财会人员/助理',
         '专业技术人员':'专业技术人员/设计师/工程师',
         '设计师':'专业技术人员/设计师/工程师',
         '工程师':'专业技术人员/设计师/工程师',
         '医护人员': '医护人员/教师/军人/科级以上',
         '科级以上': '医护人员/教师/军人/科级以上',
         '教师': '医护人员/教师/军人/科级以上',
         '军人': '医护人员/教师/军人/科级以上',
         # '销售人员': '销售人员',
         '一般科员' :'一般科员/专员',
         '专员' :'一般科员/专员',
         # '农林牧渔养殖人员' :  '农林牧渔养殖人员',
         # '演员/运动员':'演员/运动员',
         # '其他' :'其他'
    }


}




def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)  #计算时间跨度
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x


def ModifyDf(x, new_value):
    if np.isnan(x):
        return new_value
    else:
        return x

