import os
import sys
bw = int(sys.argv[1])
times = float(sys.argv[2])
next_bw = int(bw*times)
time_change = int(sys.argv[3])
with open(os.path.join("wired"+str(bw)+'-x'+str(int(times))+'-'+str(time_change)), 'w') as f:
    f.write("#"+str(1)+" "+str(bw)+"\n")
    if times == 1:
        send_pkts = bw // 12
        for j in range(send_pkts) :
            f.write(str(1)+'\n')
    else:
        f.write("#"+str(time_change*1000)+" "+str(next_bw)+"\n")
        for i in range(1, time_change*1000):
            send_pkts = bw // 12
            for j in range(send_pkts) :
                f.write(str(i)+'\n')
        for i in range(time_change*1000, 80000):
            send_pkts = next_bw // 12
            for j in range(send_pkts):
                f.write(str(i)+'\n')