import numpy as np

filename = "/home/amax/DeepLearning/SubCNN1/fast-rcnn/experiments/logs/kitti_val_caffenet_rpn_num20000.txt.2016-11-20_13-52-50"

totaltime = 0.0
num = 0.0
with open(filename) as f:

    for line in f.readlines()[799:(799+3799*2):2]:
        words = line.split()
        totaltime = totaltime+float(words[5][:-1])        
        #print "totaltime ", totaltime
        #assert(1==0)
        num = num + 1

avg_time = totaltime / num
print "avg_time ", avg_time
print "totaltime ", totaltime
print "num ", num
