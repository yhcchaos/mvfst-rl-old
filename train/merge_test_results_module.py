import os
import logging
logging.basicConfig(level=logging.ERROR)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import datetime

def fun(image_data, ax):
    #x0, y0, x1, y1 = ax.get_position().bounds
    # 显示图像并确保占满整个子图区域
    img_height, img_width = image_data.shape[:2]

    # 设置 extent，确保图像保持原始长宽比
    ax.imshow(image_data, extent=(0, img_width, 0, img_height))
    ax.axis('off')
    
def merge_test_results(cc_scheme, base_folder, num_columns, fig_col, fig_row, dpi):
    begin_time = datetime.datetime.now()
    # 获取所有子文件夹
    subfolders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    # 排序子文件夹，确保按照顺序处理
    subfolders.sort()
    num_results = len(subfolders)
    num_figures = num_results // 40 if num_results % 40 == 0 else num_results // 40 + 1
    print("num results: {}".format(num_results))
    print("figures count: {}".format(num_figures))

    rows = 40 // num_columns
    fig, axes = plt.subplots(rows, num_columns, figsize=(fig_col * num_columns, fig_row * rows))
    for row, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(base_folder, subfolder)
        throughput_images_path = []
        delay_images_path   = []
        thrroughput_images = []
        delay_images = []
        ax_row = row % 40 // num_columns
        ax_col = row % num_columns
        for run_id in range(3):
            throughput_images_path.append(os.path.join(subfolder_path, cc_scheme, '{}_datalink_throughput_run{}.png'.format(cc_scheme, run_id+1)))
            delay_images_path.append(os.path.join(subfolder_path, cc_scheme, '{}_datalink_delay_run{}.png'.format(cc_scheme, run_id+1)))
        for throughput_path, delay_path in zip(throughput_images_path, delay_images_path):
            if os.path.exists(throughput_path):
                thrroughput_images.append(plt.imread(throughput_path))
                delay_images.append(plt.imread(delay_path))
        tp_axs, delay_axs = [], []
        h = str(int(1/3*100)) + "%"
        if(len(thrroughput_images)==3):
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper left'))
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='center left'))
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='lower left'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper right'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='center right'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='lower right'))
            for tp_img, tp_axs, delay_img, delay_axs in zip(thrroughput_images, tp_axs, delay_images, delay_axs):
                fun(tp_img, tp_axs)
                fun(delay_img, delay_axs)
        elif(len(thrroughput_images)==2):
            #h = str(int(1/2*100)) + "%" # 图片高度
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper left'))
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='center left'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper right'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='center right'))
            for tp_img, tp_axs, delay_img, delay_axs in zip(thrroughput_images, tp_axs, delay_images, delay_axs):
                fun(tp_img, tp_axs)
                fun(delay_img, delay_axs)
        elif(len(thrroughput_images)==1):
            #h = str(100) + "%" # 图片高度
            tp_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper left'))
            delay_axs.append(inset_axes(axes[ax_row, ax_col], width="50%", height=h, loc='upper right'))
            for tp_img, tp_axs, delay_img, delay_axs in zip(thrroughput_images, tp_axs, delay_images, delay_axs):
                fun(tp_img, tp_axs)
                fun(delay_img, delay_axs)
        else:
            print("no images in {}".format(subfolder_path))
        axes[ax_row, ax_col].axis('off')
        axes[ax_row, ax_col].set_title(subfolder, fontsize=24)  # 设置子图标题为子文件夹名字
        rect = plt.Rectangle((0, 0), 1, 1, transform=axes[ax_row, ax_col].transAxes, color='red', linewidth=3, fill=False)
        axes[ax_row, ax_col].add_patch(rect) 
        plt.subplots_adjust(left=0, right=1, top=0.98, bottom=0, wspace=0, hspace=0.05)#plt.tight_layout()  # 自动调整子图布局
        if((row+1) % 40 == 0 or row == num_results-1):
            print("merge results {}".format(row+1))
            process_time = datetime.datetime.now()
            print("results {} are merged, start to save the big figure, use time: {}".format(row+1, process_time - begin_time))
            plt.savefig(os.path.join(base_folder, os.pardir, 'output_image{}.png'.format(row+1)), dpi= dpi)  # 保存最终的图片
            save_time = datetime.datetime.now()
            print("save the big figure done, use time: {}".format(save_time - process_time))
            fig, axes = plt.subplots(rows, num_columns, figsize=(fig_col * num_columns, fig_row * rows))
            begin_time = datetime.datetime.now()