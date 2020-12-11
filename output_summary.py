import argparse
import os
import glob
import csv
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_auc(log_path):
    with open(log_path, 'r') as f:
        auc_txt = f.read().split()
        auc = auc_txt[-1]
        fpr_at_tpr_equal_one = auc_txt[-3]
    return auc


def log_agg(output_dir='./output/mvtec/'):
    dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    dirs.sort()
    file = open(os.path.join(output_dir, 'auc.csv'), 'w')
    writer = csv.writer(file)
    writer.writerow(['dataset', 'model', 'Loss', 'score', 'bottle_ch', 'epochs', 'patch', 'mask_category', 'mask_color', 'mask_size', 'ratio', 'noise_sigma','mode', 'AUC', 'fpr@tpr=1.0', 'dir'])
    try:
        for dir in dirs:
            log_path = glob.glob(os.path.join(output_dir, dir, '*.log'))
            # print(log_path)
            for log in log_path:
                with open(log, 'r') as f:
                    auc_txt = f.read().split()
                    mode = os.path.splitext(os.path.basename(log))[0]
                    conf = re.split('[-]', dir)
                    mask_op = re.split('[_]', conf[7])
                    if len(mask_op)<3:
                        mask_op = ['None']*3
                    writer.writerow([conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6], mask_op[0], mask_op[1], mask_op[2], mask_op[3], mask_op[4], mode, auc_txt[-1], auc_txt[-3], dir])
                    print(conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], auc_txt[-2], auc_txt[-1])
    finally:
        file.close()


def output_image_sort(output_dir='./output/mvtec/'):
    for i in range(79):
        os.makedirs(os.path.join(output_dir, 'summary', 'training-{}'.format(i)), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'summary', 'roc'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'summary', 'hist'), exist_ok=True)

    dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    dirs.sort()
    for dir in dirs:
        test_path = glob.glob(os.path.join(output_dir, dir, 'test'))
        test_path = [d for d in test_path if os.path.isdir(d)]
        eval_path = glob.glob(os.path.join(output_dir, dir, 'eval*'))
        eval_path = [d for d in eval_path if os.path.isdir(d)]
        for test_dir in test_path:
            files = glob.glob(os.path.join(test_dir, '*.png'))
            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                if 'hist' not in filename and 'roc' not in filename:
                    shutil.copy2(file, os.path.join(output_dir, 'summary', filename, dir+'.png'))
                elif 'hist' in filename:
                    if 'by-class' in filename:
                        shutil.copy2(file, os.path.join(output_dir, 'summary', 'hist', dir+'_by-class.png'))
                    else:
                        shutil.copy2(file, os.path.join(output_dir, 'summary', 'hist', dir+'.png'))
                elif 'roc' in filename:
                    shutil.copy2(file, os.path.join(output_dir, 'summary', 'roc', dir+'.png'))
        for eval_dir in eval_path:
            files = glob.glob(os.path.join(eval_dir, '*.png'))
            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                if 'hist' not in filename and 'roc' not in filename:
                    if 'noliquid' not in eval_dir:
                        shutil.copy2(file, os.path.join(output_dir, 'summary', filename, dir+os.path.basename(eval_dir)+'.png'))
                    else:
                        pass
                elif 'hist' in filename:
                    shutil.copy2(file, os.path.join(output_dir, 'summary', 'hist', dir+os.path.basename(eval_dir)+'.png'))
                elif 'roc' in filename:
                    shutil.copy2(file, os.path.join(output_dir, 'summary', 'roc', dir+os.path.basename(eval_dir)+'.png'))

                # print(os.path.basename(dir))

def output_socre_sort(output_dir='./output_logs/test/'):
    score_file = open(os.path.join(output_dir, 'score.csv'), 'w')
    writer = csv.writer(score_file)
    writer.writerow(['dir', 'auc', 'normal_avg', 'normal_var', 'normal_max','anomaly_avg', 'anomaly_var'])

    dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    dirs.sort()
    try:
        for dir in dirs:
            if 'summary' in dir:
                continue
            log_path = glob.glob(os.path.join(output_dir, dir, '*.log'))
            auc = get_auc(log_path[-1])
            test_path = glob.glob(os.path.join(output_dir, dir, 'test'))
            test_path = [d for d in test_path if os.path.isdir(d)]
            for test_dir in test_path:
                files = glob.glob(os.path.join(test_dir, '*anomaly_score*'))
                for file in files:
                    df = pd.read_csv(file)
                    # print(df, file)
                    if 'index' not in df.columns.values:
                        df.insert(0, 'index', range(len(df)))
                        df.to_csv(file, index=True)
                    df_normal = df[df['anomaly_label'] == False]
                    df_anomaly = df[df['anomaly_label'] == True]
                    normal_avg = df_normal.mean().loc['anomaly_score']
                    normal_var = df_normal.var(ddof=0).loc['anomaly_score']
                    anomaly_avg = df_anomaly.mean().loc['anomaly_score']
                    anomaly_var = df_anomaly.var(ddof=0).loc['anomaly_score']

                    th = df_normal.max().loc['anomaly_score']

                    writer.writerow([dir, auc, normal_avg, normal_var, th, anomaly_avg, anomaly_var])
                    print(normal_avg, normal_var, anomaly_avg, anomaly_var)
    finally:
        score_file.close()


def create_heatmap(output_dir='./output/mvtec/'):

    data_dir = '/home/inagaki/workspace/UTAD/datasets/mvtec/wood256/test/'
    test_classes = os.listdir(data_dir)
    test_classes.pop(test_classes.index('good'))
    test_classes.insert(0, 'good')

    dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    dirs.sort()

    column_labels = []
    row_labels = []
    heatmap_data = np.zeros([0])

    for dir in dirs:
        if 'summary' in dir:
            continue
        test_path = glob.glob(os.path.join(output_dir, dir, 'test'))
        test_path = [d for d in test_path if os.path.isdir(d)]

        conf = re.split('[-]', dir)
        column_labels.append(conf[7]) # mask_option
        
        for test_dir in test_path:
            files = glob.glob(os.path.join(test_dir, '*anomaly_score*'))
            for file in files:
                df = pd.read_csv(file)
                df_normal = df[df['anomaly_label'] == False]
                df_anomaly = df[df['anomaly_label'] == True]

                th = df_normal.max().loc['anomaly_score']
                not_separated_anomaly = df_anomaly[df_anomaly['anomaly_score'] < th]
                non_separate = ''
                
                heatmap_array = df['anomaly_score'].values
                heatmap_array = np.where(heatmap_array<=th, 1, 0)
                # heatmap_array = np.where()

                if not row_labels:
                    heatmap_data = heatmap_array
                    for row in df.itertuples():
                        index = row.index
                        category_label = row.labels
                        category_first_index = df[df['labels']==category_label].min().loc['index']
                        category_index = int(index - category_first_index)
                        row_labels.append(f'{index}:{test_classes[category_label]}{category_index}')
                else:
                    heatmap_data = np.vstack([heatmap_data, heatmap_array])
    # print(row_labels)
    # print(column_labels)
    # print(heatmap_data)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(heatmap_data, cmap=plt.cm.Oranges, edgecolor='k')
    ax.set_xticks(np.arange(heatmap_data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(heatmap_data.shape[0])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.xticks(rotation=70)
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    fig.set_size_inches(20, 10)
    plt.savefig(os.path.join(output_dir, 'separatable.png'), transparent=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output/mvtec/')
    args = parser.parse_args()
    
    log_agg(args.output_dir)
    output_image_sort(args.output_dir)
    output_socre_sort(args.output_dir)
    create_heatmap(args.output_dir)

    # output_socre_sort('./output_logs/test/')
    # create_heatmap('./output_logs/test/')