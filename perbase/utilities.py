import os
import shutil
import pickle
import torch
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def build_probscores_df(ids, prob_scores, base_pos=None):

    prob_scores_dict = {}
    for i in range(prob_scores.shape[-1]): # -1 最后一个维度
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    if not base_pos:
        df_dict = {
            'id': ids
        }
    else:
        df_dict = {
            'id': ids,
            'base_pos': base_pos
        }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    # predictions_df.set_index('id', inplace=True)
    return predictions_df

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

class ReaderWriter(object):
    """用于转储、读取和记录数据的类"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """通过pickle转储数据
           参数:
               data: 要pickle的数据
               file_name: 数据将被转储的文件路径
               mode: 指定写入选项，即二进制或Unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """读取转储/反pickle的数据
           参数:
               file_name: 数据将被转储的文件路径
               mode: 指定读取选项，即二进制或Unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        使用PyTorch的自定义序列化转储张量。以便稍后在特定的gpu上重新加载张量。
        参数:
            data: 张量
            file_name: 数据将被转储的文件路径
        返回:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """读取转储/反pickle的数据
           参数:
               file_name: 数据将被转储的文件路径
               device: 加载张量到的gpu
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """将数据写入文件
           参数:
               line: 表示要写入的数据的字符串
               outfile: 数据将被写入/记录的文件路径
               mode: 指定写入选项，即追加或写入
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """将数据写入文件
           参数:
               line: 表示要写入的数据的字符串
               outfile: 数据将被写入/记录的文件路径
               mode: 指定写入选项，即追加或写入
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line


def create_directory(folder_name, directory="current"):
    """创建目录/文件夹（如果不存在）并返回目录的路径
       参数:
           folder_name: 表示要创建的文件夹的名称的字符串
       关键参数:
           directory: 表示要在其中创建文件夹的目录的字符串
                      如果为“current”，则文件夹将在当前目录中创建
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__指的是utilities.py
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('可用的GPU数量:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, 名称:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("没有可用的GPU设备！！")

def get_cuda_device_stats(device):
    print('可用的总内存:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('设备上分配的内存总量:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('设备上分配的最大内存:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('设备上缓存的总内存:', torch.cuda.memory_reserved(device)/(1024**3), 'GB')
    print('设备上缓存的最大内存:', torch.cuda.max_memory_reserved(device)/(1024**3), 'GB')


def check_na(df):
    assert df.isna().any().sum() == 0

def plot_xy(x, y, xlabel, ylabel, legend, fname, wrk_dir):
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend([legend])
    plt.savefig(os.path.join(wrk_dir, os.path.join(fname + ".pdf")))
    plt.close()

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)
