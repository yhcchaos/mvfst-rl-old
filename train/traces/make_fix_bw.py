import os
import sys
bandwidths = [12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192]
for bw in bandwidths:
    with open(os.path.join("fix", "{}mbps.trace".format(bw)), 'w') as f:
        f.write("#"+str(1)+" "+str(bw)+"\n")
        send_packets = bw // 12
        for j in range(send_packets):
            f.write(str(1)+"\n")