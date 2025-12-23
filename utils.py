import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # 增加自环
    D = np.array(np.sum(A, axis=1)).reshape((-1,))  # 计算节点度
    D[D <= 10e-5] = 10e-5  # Prevent infs，
    # 使用 10e-5 替换所有<= 10e-5 的度数，目的是防止在后续计算中出现由于某些节点的度数为0导致的分母为0而引起的无穷大(inf)的情况，。
    diag = np.reciprocal(np.sqrt(D))  # 计算规范化系数
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    # 对称规格化
    return A_wave

def get_normalized_data(X):
    X = X.transpose((1, 2, 0))  # (34272, 207, 2)-->(207, 2, 34272)第1维是以天为单位计算每个时间戳相对于当天零点的时间差，和数据内容没关系，第0维才是原数据
    X = X.astype(np.float32)
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)
    return X, means, stds

def generate_dataset(X, his_num, pred_num, means, stds):
    # X(207, 2, 34272)
    indices = [(i, i + (his_num + pred_num)) for i in range(X.shape[2] - (his_num + pred_num) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i: i + his_num].transpose((0, 2, 1)))
        # features(207, i: i + his_num, 2)
        target.append(X[:, 0, i + his_num: j] * stds[0] + means[0])
        # target(207, 1(only flow), i + num_time steps_input: j)
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))

def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)
    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)

        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE

    return result

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mask_metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)
    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        MSE = masked_mse(pred_i, y_i, 0.0)
        RMSE = masked_rmse(pred_i, y_i, 0.0)
        MAE = masked_mae(pred_i, y_i, 0.0)
        MAPE = masked_mape(pred_i, y_i, 0.0)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE

    return result

def result_print(result, info_name='Evaluate', test_dataset=None):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
    print("========== {} results ==========".format(info_name))
    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
    print("---------------------------------------")

    if info_name == 'Best':
        print("========== Best results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print("---------------------------------------")

    with open('output.txt','a') as file:
        file.write("========== {} results of {}========== \n".format(info_name, test_dataset))
        file.write(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f \n"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        file.write("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f \n"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        file.write("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f \n"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        file.write("--------------------------------------- \n")