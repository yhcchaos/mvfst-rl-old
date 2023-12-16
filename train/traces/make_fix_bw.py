import os
import sys
bandwidths = [36, 60, 72, 84, 120, 144, 168, 240, 288, 336]
for bw in bandwidths:
    with open(os.path.join("fix", "{}mbps.trace".format(bw)), 'w') as f:
        f.write("#"+str(1)+" "+str(bw)+"\n")
        send_packets = bw // 12
        for j in range(send_packets):
            f.write(str(1)+"\n")
                