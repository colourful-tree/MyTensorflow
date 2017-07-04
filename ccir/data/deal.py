# encoding: UTF-8
import re
def strQ2B(s):
    n = []
    s = s.decode('utf-8')
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 == num or 0xFF03 <= num < 0xFF3C or 0xFF3C < num <= 0xFF5E:
            num -= 0xfee0
        num = unichr(num)
        n.append(num)
    return ''.join(n) 

#with open("json.longshort.new.all","r") as fin:
with open("json.testset","r") as fin:
    #with open("ccir.json","w") as fout:
    with open("ccir.test.json","w") as fout:
        for i in fin:
            i = strQ2B(i)
            i = i.replace("<br>","")
            #i = i.replace("　","")
            #i = re.sub(r"．+",".", i)
            i = re.sub(r"\.+",".", i)
            i = re.sub(r"-+",",", i)
            i = re.sub(r"~+",",", i)
            i = re.sub("[\！\/$%^*(+\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf-8"),"",i)
            fout.write(i.encode("utf-8")+",")
