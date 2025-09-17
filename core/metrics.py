import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error


def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


def tensor2allcsv(visuals, col_num):
    df = pd.DataFrame()
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    inf_df = pd.DataFrame(squeeze_tensor(visuals['INF']))

    if col_num != 1:
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(labels=i, axis=1, inplace=True)
            ori_df.drop(labels=i, axis=1, inplace=True)
            lr_df.drop(labels=i, axis=1, inplace=True)
            inf_df.drop(labels=i, axis=1, inplace=True)

    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    df['INF'] = inf_df.mean(axis=1)

    df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['label'] = squeeze_tensor(visuals['label'])

    differ_df = (sr_df - ori_df)

    return df, sr_df, differ_df


def merge_all_csv(all_datas, all_data):
    all_datas = pd.concat([all_datas, all_data])
    return all_datas


def save_csv(data, data_path):
    data.to_csv(data_path, index=False)


def get_mean(df):
    mean = df['value'].astype('float32').mean()
    normal_mean = df['value'][df['label'] == 0].astype('float32').mean()
    anomaly_mean = df['value'][df['label'] == 1].astype('float32').mean()

    return mean, normal_mean, anomaly_mean


def get_val_mean(df):
    mean_dict = {}

    ori = 'ORI'
    ori_mean = df[ori].astype('float32').mean()
    ori_normal_mean = df[ori][df['label'] == 0].astype('float32').mean()
    ori_anomaly_mean = df[ori][df['label'] == 1].astype('float32').mean()

    gen_mean = df['SR'].astype('float32').mean()
    gen_normal_mean = df['SR'][df['label'] == 0].astype('float32').mean()
    gen_anomaly_mean = df['SR'][df['label'] == 1].astype('float32').mean()

    mean_dict['MSE'] = mean_squared_error(df[ori], df['SR'])

    mean_dict['ori_mean'] = ori_mean
    mean_dict['ori_normal_mean'] = ori_normal_mean
    mean_dict['ori_anomaly_mean'] = ori_anomaly_mean

    mean_dict['gen_mean'] = gen_mean
    mean_dict['gen_normal_mean'] = gen_normal_mean
    mean_dict['gen_anomaly_mean'] = gen_anomaly_mean

    mean_dict['mean_differ'] = ori_mean - gen_mean
    mean_dict['normal_mean_differ'] = ori_normal_mean - gen_normal_mean
    mean_dict['anomaly_mean_differ'] = ori_anomaly_mean - gen_anomaly_mean

    mean_dict['ori_no-ano_differ'] = ori_normal_mean - ori_anomaly_mean
    mean_dict['ori_mean-no_differ'] = ori_mean - ori_normal_mean
    mean_dict['ori_mean-ano_differ'] = ori_mean - ori_anomaly_mean

    mean_dict['gen_no-ano_differ'] = gen_normal_mean - gen_anomaly_mean
    mean_dict['gen_mean-no_differ'] = gen_mean - gen_normal_mean
    mean_dict['gen_mean-ano_differ'] = gen_mean - gen_anomaly_mean

    return mean_dict

#
def relabeling_strategy(df, params):
    y_true = []
    best_N = 0
    best_f1 = -1
    best_thred = 0
    best_predictions = []
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label'])

    df_sort = df.sort_values(by="differ", ascending=False)
    df_sort = df_sort.reset_index(drop=False)

    for t in thresholds:
        # if (t - 1) % params['step_t'] == 0:
        #     print("t: ", t)
        y_true, y_pred, thred = predict_labels(df_sort, t)
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                j = i - 1
                while j >= 0 and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j -= 1
                j = i + 1
                while j < len(y_pred) and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j += 1

        f1 = calculate_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_N = t
            best_thred = thred
            best_predictions = y_pred

    accuracy = calculate_accuracy(y_true, best_predictions)
    precision = calculate_precision(y_true, best_predictions)
    recall = calculate_recall(y_true, best_predictions)

    return best_f1,precision,recall

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def relabeling_strategy_fast(df, params):
    """
    假设:
      - df 包含真值列 'label'（0/1）
      - relabel 的策略是选择排序后前 t 个为正例（top-N）
      - thresholds 是整数 (top-N)，若不是请参考下方说明
    """
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label']).astype(int)
    df_sort = df.sort_values(by="differ", ascending=False).reset_index(drop=True)
    n = len(df_sort)
    y_true = df_sort['label'].to_numpy().astype(np.uint8)

    # 预计算连续为1的段（runs），并为每个位置标记它属于哪个 run（或 -1）
    mask = (y_true == 1)
    run_id = np.full(n, -1, dtype=int)
    runs = []  # list of (start, end)
    run_idx = 0
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append((i, j))
        run_id[i:j+1] = run_idx
        run_idx += 1
        i = j + 1

    run_marked = np.zeros(len(runs), dtype=bool)  # 某个 run 是否已经被整个标为1
    y_pred = np.zeros(n, dtype=np.uint8)  # 当前预测（随 t 增长不断更新）
    best_f1 = -1.0
    best_preds = None
    best_t = None

    # thresholds 应该单调增；我们逐步把前 t 的点置 1（增量更新）
    # 为了兼容 thresholds 任意跳变，先排序唯一化并遍历
    thresholds_sorted = np.unique(np.sort(thresholds))
    cur_t = 0
    for t in thresholds_sorted:
        if t > n:
            t = n
        # 增量把 index cur_t .. t-1 加入预测（这些 index 是按 differ 排序的）
        while cur_t < t:
            pos = cur_t  # 因为 df_sort 已按 differ desc 排序，top-N 是前 N 的位置
            rid = run_id[pos]
            if rid != -1 and not run_marked[rid]:
                s, e = runs[rid]
                y_pred[s:e+1] = 1
                run_marked[rid] = True
            else:
                # 不属于任何 run，或 run 已标过，直接标记该点（若 run 已标过 slice 已为1）
                y_pred[pos] = 1
            cur_t += 1

        # 计算指标（使用 sklearn 的 C 实现，速度快）
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = y_pred.copy()
            best_t = t

    # 最后的 accuracy/precision/recall
    acc = accuracy_score(y_true, best_preds)
    prec = precision_score(y_true, best_preds, zero_division=0)
    rec = recall_score(y_true, best_preds, zero_division=0)

    return best_f1,prec,rec
    # return {
    #     'best_f1': best_f1,
    #     'best_t': int(best_t),
    #     'best_predictions': best_preds,
    #     'accuracy': float(acc),
    #     'precision': float(prec),
    #     'recall': float(rec)
    # }


#
# def relabeling_strategy(df, params):
#     thresholds = np.arange(params['start_label'],
#                            params['end_label'],
#                            params['step_label'])
#
#     df_sort = df.sort_values(by="differ", ascending=False).reset_index(drop=True)
#
#     best_result = {
#         "f1": -1,
#         "threshold": None,
#         "t_value": None,
#         "predictions": None,
#         "y_true": None
#     }
#
#     for t in thresholds:
#         y_true, y_pred, thred = predict_labels(df_sort, t)
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#
#         # ---- 向量化修正预测 ----
#         # 找出 y_true == 1 的连续区间
#         mask = y_true == 1
#         diff = np.diff(mask.astype(int), prepend=0, append=0)
#         starts = np.where(diff == 1)[0]
#         ends   = np.where(diff == -1)[0]
#
#         # 遍历每个正例区间
#         for s, e in zip(starts, ends):
#             if np.any(y_pred[s:e] == 1):   # 只要该区间有一个预测命中
#                 y_pred[s:e] = 1           # 整个区间补齐
#
#         # ---- 计算指标 ----
#         f1 = calculate_f1(y_true, y_pred)
#
#         if f1 > best_result["f1"]:
#             best_result.update({
#                 "f1": f1,
#                 "threshold": thred,
#                 "t_value": t,
#                 "predictions": y_pred.copy(),
#                 "y_true": y_true.copy()
#             })
#
#     # 在最佳结果下计算其他指标
#     y_true, y_pred = best_result["y_true"], best_result["predictions"]
#     metrics = {
#         "f1": best_result["f1"],
#         "accuracy": calculate_accuracy(y_true, y_pred),
#         "precision": calculate_precision(y_true, y_pred),
#         "recall": calculate_recall(y_true, y_pred),
#         "best_threshold": best_result["threshold"],
#         "best_t": best_result["t_value"]
#     }
#     return metrics

def predict_labels(df_sort, num):
    df_sort['pred_label'] = 0
    df_sort.loc[0:num - 1, 'pred_label'] = 1
    thred = df_sort.loc[num - 1, 'differ']

    df_sort = df_sort.set_index('index')
    df_sort = df_sort.sort_index()

    y_true = df_sort['label'].tolist()
    y_pred = df_sort['pred_label'].tolist()

    return y_true, y_pred, thred


def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def calculate_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision


def calculate_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall


def calculate_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1
