"""
从 www.a-hospital.com 读取正常医院的名字
不正常医院的名字从 https://github.com/shenwei356/BlackheartedHospital 获取
"""
import requests
from bs4 import BeautifulSoup
import json
host = "http://www.a-hospital.com"
base_url = host + "/w/%E5%85%A8%E5%9B%BD%E5%8C%BB%E9%99%A2%E5%88%97%E8%A1%A8"
soup = BeautifulSoup(requests.get(base_url).content,"lxml")
province_urls = [host + s.get("href") for s in soup.find(name="map",attrs={"name":"ImageMap_1_1550133785","id":"ImageMap_1_1550133785"}).find_all(name="area")]
normal_hospital_list = []
for province_url in province_urls:
    normal_hospital_list += [s for s in [x.get("title") for x in BeautifulSoup(requests.get(
        province_url
    ).content,"lxml").find(name="div",attrs={"id":"bodyContent"}).find("ul").find_all("a")] if s is not None]
print(len(normal_hospital_list))
with open("normal_list.txt","w") as fp:
    fp.write("\n".join(normal_hospital_list))