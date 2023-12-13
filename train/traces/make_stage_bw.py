import os
with open(os.path.join("stage", "24mbps.trace"), 'w') as f:
    bandwidth = [12, 36]
    start_time = 1
    for j in range (2):
        for bw in bandwidth:
            send_packets = bw // 12
            f.write("#"+str(bw)+"\n")
            for i in range(30*1000):
                time = start_time + i
                for j in range(send_packets):
                    f.write(str(time)+"\n")
            start_time = start_time + 30*1000        