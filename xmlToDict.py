# -*- coding: utf-8 -*-
from xml2dict import XML2Dict


def XmlTodict(xmlFilePath):
    xml = XML2Dict()
    r = xml.parse(xmlFilePath)

    dic = dict()
    for item in r.annotation.object:
        dic.setdefault(item.name, []).append(int(item.bndbox.xmin))
        dic.setdefault(item.name, []).append(int(item.bndbox.ymin))
        dic.setdefault(item.name, []).append(int(item.bndbox.xmax) - int(item.bndbox.xmin))
        dic.setdefault(item.name, []).append(int(item.bndbox.ymax) - int(item.bndbox.ymin))

    # print(dic)
    return dic

# print(XmlTodict('ModeLabel_00001.xml'))
