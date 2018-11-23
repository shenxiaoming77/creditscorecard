import requests
from lxml import etree
import re
from urllib.request import urlretrieve
#1 获取视频id
#2 拼接完整URL
#3 获取视频播放地址
#4 下载视频

def download(url):
    # url='http://www.pearvideo.com/category_9'
    #获取页面源代码
    response = requests.get(url)
    if response.status_code == 200:  # 确认状态码为200
        html =  response.text  # 如果成立就返回页面源代码，有误就是返回空
    else:
        print('none response, url: ', url)
        return None

    #把文本文件处理成可解析的对象
    html=etree.HTML(html)
    video_id=html.xpath('//div[@class="vervideo-bd"]/a/@href')

    # 获取页面中所有的div//
    #列表
    video_url=[]
    starturl = 'http://www.pearvideo.com'

    # 拼接完整url
    for id in video_id:
        newurl=starturl+'/'+id
        video_url.append(newurl)

    # 获取视频播放地址
    for playurl in video_url:
        # 获取页面源代码
        html=requests.get(playurl).text
        # 正则匹配 .*?匹配所有
        req='srcUrl="(.*?)"'
        # req=re.compile(req)
        # 视频真正的播放url
        purl=re.findall(req, html)
        # print(purl)
        print(purl[0])
        # 获取视频名称
        req = '<h1 class="video-tt">(.*?)</h1>'
        pname=re.findall(req, html)
        title = pname[0]

        #获取视频的summary
        req = '<></h1>'

        print("正在下载视频:%s"%pname[0])
        # 下载的url 下载的地址
        return
        try:
            urlretrieve(purl[0], 'video/%s.mp4' % pname[0])
        except OSError as err:
            print('exception..............')
            print(err)

def downloadmore():
    n=12
    while True:
        if n > 48:
            # 跳出循环的
            return
        url = "http://www.pearvideo.com/category_loading.jsp?reqType=5&categoryId=9&start=%d"%n
        print('download url:')
        print(url)
        n += 12
        download(url)
downloadmore()