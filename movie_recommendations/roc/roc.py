def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

with open("res","r") as fin:
    pre = [float(i) for i in fin.readline().strip().split(",")]
    label = [int(i) for i in fin.readline().strip().split(",")]

cnt = len(pre)

l = []
p = []
for i in range(cnt):
    if pre[i] > 2 and pre[i] < 4:
        p.append(pre[i])
        l.append(label[i])
roc = []
for th in floatrange(2,4,101):
    right = 0
    error = 0
    right_total = 0
    error_total = 0
    for i in range(len(l)):
        if l[i] == 3:
            right_total += 1
        else:
            error_total += 1
        if l[i] == 3 and p[i] < th:
            right += 1
            continue
        if l[i] != 3 and p[i] < th:
            error += 1
            continue
    roc.append([error*1.0/error_total, 
                right*1.0/right_total])
for i in roc:
    print str(i[0])+"\t"+str(i[1])
#print l
#print p
