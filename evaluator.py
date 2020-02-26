import numpy as np
import math
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('groundtruth', help='ground truth file')
parser.add_argument('csvfile', help='experiment csv file')
args = parser.parse_args()

gtrows=[]
exrows=[]

with open(args.groundtruth, newline='') as gtfile:
    gtreader = csv.reader(gtfile, delimiter=',')
    for gtrow in gtreader:
        gtrows.append((gtrow[0],int(gtrow[1]), int(gtrow[2]), int(gtrow[3])))
with open(args.csvfile, newline='') as exfile:
    exreader = csv.reader(exfile, delimiter=',')
    for exrow in exreader:
        exrows.append((exrow[0],int(exrow[1]), int(exrow[2]), int(exrow[3])))

for exrow in exrows:
    for gtrow in gtrows:
        if exrow[0] == gtrow[0]:
            pass
            #print((exrow[0],exrow[1]-gtrow[1],exrow[2]-gtrow[2],exrow[3]-gtrow[3]))

gdeltas = []
for (gtrow1,gtrow2) in zip(gtrows,gtrows[1:]):
    gdeltas.append((gtrow2[0],gtrow2[1]-gtrow1[1],gtrow2[2]-gtrow1[2],gtrow2[3]-gtrow1[3]))

edeltas = []
for (exrow1,exrow2) in zip(exrows,exrows[1:]):
    edeltas.append((exrow2[0],exrow2[1]-exrow1[1],exrow2[2]-exrow1[2],exrow2[3]-exrow1[3]))

#print(gdeltas,edeltas)

egdeltas=[]
for edelta in edeltas:
    for gdelta in gdeltas:
        if gdelta[0] == edelta[0]:
            print((edelta[0], edelta[1:4], gdelta[1:4]))
            egdeltas.append((edelta[0], edelta[1:4], gdelta[1:4]))
#print(egdeltas)

scores = []
dots = []
mage = []
magg = []

u = []
v = []

for egdelta in egdeltas:
    e = egdelta[1]
    g = egdelta[2]
    #print((egdelta[0], e[0] - g[0], e[1] - g[1], e[2] - g[2]))
    u.append(e[0])
    u.append(e[2])
    v.append(g[0])
    v.append(g[2])
    dots.append(e[0]*g[0])
    dots.append(e[2]*g[2])
    mage.append(e[0]*e[0]+e[2]*e[2])
    magg.append(g[0]*g[0]+g[2]*g[2])

#print(dots,mage,magg)

def cos_sim(u, v):
    u = np.array(u)
    v = np.array(v)
    return np.dot(u, v) / (np.sqrt(np.dot(u,u)) * np.sqrt(np.dot(v,v)))

print(sum(dots) / math.sqrt(sum(mage)*sum(magg)))
# print(cos_sim(u,v))

