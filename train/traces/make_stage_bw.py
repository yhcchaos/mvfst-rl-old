import os
import sys
bw = int(sys.argv[1])
next_bw = int(sys.argv[2])
time_change = int(sys.argv[3])
with open(os.path.join("stage", str(bw)+'-'+str(next_bw)+'-'+str(time_change)+'mbps.trace'), 'w') as f:
    f.write("#"+str(1)+" "+str(bw)+"\n")
    f.write("#"+str(time_change*1000)+" "+str(next_bw)+"\n")
    for i in range(1, time_change*1000):
        send_pkts = bw // 12
        for j in range(send_pkts) :
            f.write(str(i)+'\n')
    for i in range(time_change*1000, 40000):
        send_pkts = next_bw // 12
        for j in range(send_pkts):
            f.write(str(i)+'\n')