import os
import sys
bw = int(sys.argv[1])
with open(os.path.join("fix", "{}mbps.trace".format(bw)), 'w') as f:
    f.write("#"+str(1)+" "+str(bw)+"\n")
    send_packets = bw // 12
    for j in range(send_packets):
        f.write(str(1)+"\n")
                