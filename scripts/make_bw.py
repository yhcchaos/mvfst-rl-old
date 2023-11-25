import random
import os
import numpy as np
import argparse
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
def add_value(x, value):
    return x + value
def parse_args():
    parser = argparse.ArgumentParser(description= '给出最小带宽、最大带宽，以固定带宽间隔构造固定带宽、泊松变化带宽或阶梯变化带宽')
    sub_parser = parser.add_subparsers(dest= 'trace_type')
    fix_parser = sub_parser.add_parser('fix', help= '固定带宽')
    poi_parser = sub_parser.add_parser('poi', help= '泊松变化带宽')
    stage_parser = sub_parser.add_parser('stage', help='接替带宽')
    exp_fix_parser = sub_parser.add_parser('exp_fix', help= "指数增长固定带宽")
    exp_poi_parser = sub_parser.add_parser('exp_poi', help= "指数增长poi固定带宽")
    exp_stage_parser = sub_parser.add_parser('exp_stage', help= "指数增长poi固定带宽")
    
    for p in (fix_parser, poi_parser, stage_parser):
        p.add_argument('-a', '--min_bw', type= int, help= '最小带宽 （mbps）， 最小为1mbps')
        p.add_argument('-b', '--max_bw', type= int, help= '最大带宽（mbps）')
        p.add_argument('-i', '--bw_interval', type= int, help= "带宽的构造间隔（ms）")
        p.add_argument('--bw_array', nargs= '+', type=int, help= "带宽的数组, 指定了这个那么上面三个指定了也无效")
    
    for p in (exp_fix_parser, exp_poi_parser, exp_stage_parser):
        p.add_argument('base_bw', type= int, help= '基础带宽 （mbps）， 最小为1mbps')
        p.add_argument('exp_base', type= int, help= '倍数')
        p.add_argument('exp_times', type= int, help= '乘的次数')
    
    poi_parser.add_argument('-t', '--run_time', type= int, required= True, help= "trace持续时间（ms）")
    exp_poi_parser.add_argument('-t', '--run_time', type= int, required= True, help= "trace持续时间（ms）")
    stage_parser.add_argument('-c', '--change_duration', type= int, required= True, help= "带宽变化时间间隔（ms）,12ms的整数倍")
    exp_stage_parser.add_argument('-c', '--change_duration', type= int, required= True, help= "带宽变化时间间隔（ms）,12ms的整数倍")
    
    args = parser.parse_args()
    return args

def make_fix_bw(bw_array, trace_dir):
    for bw in bw_array:
        packet_send_time = []
        even_time_duration = 12000 / bw 
        for i in range(0, bw-1):
            if 12000 % bw != 0:
                bias = 1 if random.random() < 0.5 else 0
            else:
                bias = 0
            packet_send_time = np.append(packet_send_time, int(even_time_duration) + bias)
        packet_send_time = np.cumsum(packet_send_time)
        packet_send_time = np.append(packet_send_time, 12000)
        np.savetxt(os.path.join(trace_dir, str(bw)+"mbps.trace"), packet_send_time, fmt="%d")

def main():
    args = parse_args()
    trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'train', 'traces', args.trace_type)
    os.makedirs(trace_dir, exist_ok= True)
    #fix trace
    if args.trace_type == 'fix':
        if(args.bw_array == None):
            args.bw_array = np.arange(args.min_bw, args.max_bw+1, args.bw_interval)
        make_fix_bw(args.bw_array, trace_dir)
    #exp trace
    elif args.trace_type == 'exp':
        packet_send_time = []
        bw_array = [args.base_bw]
        for i in range(args.exp_times):
            bw_array.append(bw_array[-1] * args.exp_base)
        make_fix_bw(bw_array, trace_dir)
    #Poisson trace
    elif args.trace_type == 'poi':
        packet_send_time = []
        if(args.bw_array == None):
            args.bw_array = np.arange(args.min_bw, args.max_bw+1, args.bw_interval)
        for bw in args.bw_array:
            even_time_duration = 12000 / bw 
            packet_send_time = np.random.exponential(scale= even_time_duration, size = int(float(args.run_time) * 1000 * bw / 12000))
            packet_send_time = np.round(packet_send_time).astype(np.int)
            packet_send_time = np.cumsum(packet_send_time)
            while(packet_send_time[-1] <= args.run_time * 1000):
                next_value = packet_send_time[-1] + int(np.round(np.random.exponential(scale= even_time_duration)))
                packet_send_time = np.append(packet_send_time, next_value)
            np.savetxt(os.path.join(trace_dir, str(bw)+"mbps.trace"), packet_send_time, fmt='%d')
            packet_send_time = []
    #stage trace
    elif args.trace_type == 'stage':
        packet_send_time = []
        if(args.bw_array == None):
            args.bw_array = np.arange(args.min_bw, args.max_bw+1, args.bw_interval)
        for bw in args.bw_array:
            even_time_duration = 12000 / bw 
            if(len(packet_send_time) == 0):
                packet_send_time = np.append(packet_send_time, 0) 
            round = args.change_duration  // 12
            for r in range(round):
                send_time_in_12ms = []
                for i in range(0, bw-1):
                    if 12000 % bw != 0:
                        bias = 1 if random.random() < 0.5 else 0
                    else:
                        bias = 0
                    send_time_in_12ms = np.append(send_time_in_12ms, int(even_time_duration) + bias)
                send_time_in_12ms = np.cumsum(send_time_in_12ms)
                send_time_in_12ms = np.append(send_time_in_12ms, 12000)
                packet_send_time = np.append(packet_send_time, list(map(lambda x: add_value(x, packet_send_time[-1]), send_time_in_12ms))) 
        packet_send_time = packet_send_time[1:]
        ave_bw = len(packet_send_time) * 1500 * 8 / packet_send_time[-1]
        np.savetxt(os.path.join(trace_dir, str(ave_bw)+"mbps.trace"), packet_send_time, fmt='%d')   
    else:
        print("wrong trace type")
        
if __name__ == "__main__":
    main()


                
            