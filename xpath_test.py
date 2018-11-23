#-*- coding: UTF-8 -*-

from lxml import etree

source = u'''
<div><p class="p1" data-a="1">测试数据1</p>
<p class="p1" data-a="2">测试数据2</p>
<p class="p1" data-a="3" style="height:100px;">
<strong class="s">测试数据3</strong>
</p>
<p class="p1" data-a="4" width="200"><img src="1.jpg" class="img"/><br/>
图片</p>
'''

source2 = u'''
 <div class="share-to" id="share-to" data-type="2" data-title="独居老太病倒,消防空降喂面又梳头"
 data-summary="3月18日，贵州铜仁。消防部门接到群众报警，一位独居的55岁的老人病倒在家，
 已连续几天没有出门。消防从5楼空降入屋，给老人喂泡面，梳头。
" data-picurl="http://image2.pearvideo.com/cont/20170319/cont-1050230-10199078.png">
'''

# 从字符串解析
page = etree.HTML(source)
page2 = etree.HTML(source2)

print('元素列表：')
# 元素列表
ps = page.xpath("//p")
for p in ps:
    print (u"属性：%s" % p.attrib)
    print (u"文本：%s" % p.text)

print('文本列表：')
# 文本列表
ts = page.xpath("//p/text()")
for t in ts:
    print (t)

print('source xpath 定位：')
# xpath定位
ls = page.xpath('//p[@class="p1"][last()]/img')
for l in ls:
    print (l.attrib)

print('source2 xpath 定位:')
html.xpath('//div[@class="vervideo-bd"]/a/@href')
ls = page2.xpath('//div[@class="share-to" id="share-to" data-type="2"]')