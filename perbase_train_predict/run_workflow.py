import os
import itertools
from utilities import get_device, create_directory, ReaderWriter, perfmetric_report_categ, plot_loss
from model import Categ_CrisCasTransformer
from dataset import construct_load_dataloaders
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.multiprocessing as mp



class TrfHyperparamConfig:
    '''定义了 Transformer 模型的超参数配置'''

    def __init__(self, embed_dim, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, 
                l2_reg, batch_size, num_epochs):
        self.embed_dim = embed_dim # 64
        self.num_attn_heads = num_attn_heads # 8
        self.num_transformer_units = num_transformer_units # 2
        self.p_dropout = p_dropout # 0.1
        self.nonlin_func = nonlin_func # nn.ReLU()
        self.mlp_embed_factor = mlp_embed_factor # 2
        self.l2_reg = l2_reg # 0
        self.batch_size = batch_size # 200
        self.num_epochs = num_epochs # 20


    def __repr__(self):
        desc = " embed_dim:{}\n num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n " \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.embed_dim,
                                                                     self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc

def generate_models_config(hyperparam_config, experiment_desc, model_name, run_num, fdtype):
    '''
    根据给定的超参数配置和实验描述生成模型的配置和选项。
    注: 通用配置（generic_config）在所有模型之间共享, 保留作为占位符，以便以后传递自定义的通用配置
    '''

    dataloader_config = {'batch_size': hyperparam_config.batch_size, # 200
                         'num_workers': 0}
    generic_config = {'fdtype':fdtype, 'to_gpu':True} # torch.float32

    config = {'dataloader_config': dataloader_config, # batch_size num_workers
              'model_config': hyperparam_config, # TrfHyperparamConfig(64, 8, 2, 0.1, nn.ReLU(), 2, 0, 200, 20)
              'generic_config': generic_config # fdtype to_gpu
             }

    options = {'experiment_desc': experiment_desc, # "my_experiment"
               'run_num': run_num, # -1
               'model_name': model_name, # "Transformer"
               'num_epochs': hyperparam_config.num_epochs, # 20
               'weight_decay': hyperparam_config.l2_reg} # 0

    return config, options


def build_custom_config_map(experiment_desc, model_name):
    '''根据给定的实验描述和模型名称构建自定义配置映射'''

    if(model_name == 'Transformer'):
        # hyperparam_config = TrfHyperparamConfig(32, 8, 12, 0.3, nn.ReLU(), 2, 0, 200, 20)
        hyperparam_config = TrfHyperparamConfig(64, 8, 2, 0.1, nn.ReLU(), 2, 1e-3, 200, 20)
    run_num = -1 
    fdtype = torch.float32
    mconfig, options = generate_models_config(hyperparam_config, experiment_desc, model_name, run_num, fdtype)

    return mconfig, options

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    '''将字典内容保存到文件中'''
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

def process_multilayer_multihead_attn(attn_dict, seqs_id):
    '''处理多层多头注意力机制的注意力字典'''
    attn_dict_perseq = {}
    for l in attn_dict:
        for h in attn_dict[l]:
            tmp = attn_dict[l][h].detach().cpu()
            for count, seq_id in enumerate(seqs_id):
                if(seq_id not in attn_dict_perseq):
                    attn_dict_perseq[seq_id] = {} 
                if(l in attn_dict_perseq[seq_id]):
                    attn_dict_perseq[seq_id][l].update({h:tmp[count]})
                else:
                    attn_dict_perseq[seq_id][l] = {h:tmp[count]}
    return attn_dict_perseq

def run_categTrf(dataset_fold, dsettypes, config, options, wrk_dir, state_dict_dir=None, to_gpu=True, gpu_index=0):
    '''
    运行分类 categTransformer 模型的训练和验证
    Args:
        dataset_fold:
            {   
                0: {'train': target_id = self.partition_ids[indx]  self.criscas_datatensor[target_id],
                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                    'class_weights': tensor([0.6957, 1.7778]),},
                1: 
            }
        dsettypes: 
            ['train', 'validation']
        config: 
            {
                'dataloader_config': {'batch_size': hyperparam_config.batch_size, # 200
                                        'num_workers': 0}
                'model_config': hyperparam_config, # TrfHyperparamConfig(64, 8, 2, 0.1, nn.ReLU(), 2, 0, 200, 20)
                'generic_config': generic_config # fdtype to_gpu
            }
        options: 
            {
                'experiment_desc': experiment_desc, # "my_experiment"
                'run_num': run_num, # -1
                'model_name': model_name, # "Transformer"
                'num_epochs': hyperparam_config.num_epochs, # 20
                'weight_decay': hyperparam_config.l2_reg} # 0
        wrk_dir: 
            /Users/jarrycf/Desktop/BEtools2.0/perbase_train/ABEmax/train_val/run_0
    '''

    # 创建 dataloader
    pid = "{}".format(os.getpid()) # 获取当前进程的进程ID 10854
    dataloader_config = config['dataloader_config'] # {'batch_size': 200, 'num_workers': 0}
    cld = construct_load_dataloaders(dataset_fold, dsettypes, 'categ', dataloader_config, wrk_dir)
    data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out = cld
    '''
    data_loaders: 
    epoch_loss_avgbatc: {train: []}
    epoch_loss_avgsamples: {train: []}
    score_dict: {train: best_epoch_indx:{}\n binary_f1:{}\n macro_f1:{}\n accuracy:{}\n auc:{}\n}
    class_weight: {'train': [0.6957, 1.7778]}
    flog_out: {'train': ...perbase_train/ABEmax/train_val/run_0/train.log}
    '''

    device = get_device(to_gpu, gpu_index)  # cpu
    generic_config = config['generic_config'] # torch.float32  to_gpu
    fdtype = generic_config['fdtype'] # torch.float32

    if 'train' in class_weights: 
        # 将类别权重更新为指定的数据类型
        class_weights = class_weights['train'].type(fdtype).to(device) # [2.3655, 0.6340]
    else:
        class_weights = torch.tensor([1, 1]).type(fdtype).to(device)
    print("class weights", class_weights) # [2.3655, 0.6340]

    # 定义损失函数
    # loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='mean')
    loss_func = torch.nn.NLLLoss(reduction='mean')
    # loss_func = torch.nn.NLLLoss(reduction='mean')
    # loss_func = torch.nn.CTCLoss(reduction='mean')
    # loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='none')
    # loss_func = torch.nn.CrossEntropyLoss()
    # binary cross entropy
    # loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='mean')
    # loss_func = torch.nn.BCEWithLogitsLoss(weight=class_weights, reduction='mean')

    # 获取训练的总轮数
    num_epochs = options.get('num_epochs', 50) # 50
    # 获取当前运行的编号
    run_num = options.get('run_num') # 0

    # 获取模型的配置
    model_config = config['model_config'] # TrfHyperparamConfig(64, 8, 2, 0.1, nn.ReLU(), 2, 1e-3, 200, 20)
    # 获取模型的名称    
    model_name = options['model_name'] # Transformer

    if(model_name == 'Transformer'):
        criscas_categ_model = Categ_CrisCasTransformer(input_size=model_config.embed_dim, #64
                                                        num_nucleotides=4, 
                                                        seq_length=20, 
                                                        num_attn_heads=model_config.num_attn_heads, # 8
                                                        mlp_embed_factor=model_config.mlp_embed_factor, # 2
                                                        nonlin_func=model_config.nonlin_func, 
                                                        pdropout=model_config.p_dropout, # 0.1
                                                        num_transformer_units=model_config.num_transformer_units, # 2
                                                        pooling_mode='attn',
                                                        num_classes=2)

    # 获取模型的参数列表
    models_param = list(criscas_categ_model.parameters())
    # 将模型对象和模型名称放入列表中。
    models = [(criscas_categ_model, model_name)]

    # 如果给定了状态字典目录，则加载已保存的模型的状态字典
    if state_dict_dir:  
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # 更新模型的数据类型并移到设备上
    for model, model_name in models:
        model.type(fdtype).to(device)

    if 'train' in data_loaders:
        # 获取权重衰减
        weight_decay = options.get('weight_decay', 1e-3)
        # 定义优化器
        optimizer = torch.optim.RMSprop(models_param, weight_decay=weight_decay, lr=1e-4)
        # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
        # https://arxiv.org/pdf/1506.01186.pdf
        # pytorch version >1.1, scheduler should be called after optimizer
        # for cyclical lr scheduler, it should be called after each batch update
        # 获取训练数据加载器的迭代次
        num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
        # 计算循环学习率调度器的步长
        c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
        # 基础学习率。
        base_lr = 3e-4
        # 最大学习率
        max_lr = 5*base_lr  # 3-5 times base_lr
        # 定义循环学习率调度器
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                          mode='triangular', cycle_momentum=True)

    # 如果有验证数据，则执行以下操作。
    if ('validation' in data_loaders):
        # 创建用于存储模型状态字典的目录。
        m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1): # 如果训练次数大于1，则执行以下操作。
        # 创建用于存储损失图的目录。
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # 创建用于存储配置文件的目录。
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))

    # store attention weights for validation and test set
    # 创建存储验证和测试集注意力权重的字典。
    seqid_fattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}
    # 创建存储验证和测试集多层多头注意力权重的字典
    seqid_mlhattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}

    for epoch in range(num_epochs):
        # print("-"*35)
        for dsettype in dsettypes:
            print("device: {} | experiment_desc: {} | run_num: {} | epoch: {} | dsettype: {} | pid: {}"
                  "".format(device, options.get('experiment_desc'), run_num, epoch, dsettype, pid))
            pred_class = [] # 初始化预测类别列表。
            ref_class = [] # 初始化参考类别列表。
            prob_scores = [] # 初始化概率分数列表。
            seqs_ids_lst = [] # 初始化序列ID列表。
            data_loader = data_loaders[dsettype] # 获取当前数据集类型的数据加载器
            # total_num_samples = len(data_loader.dataset)
            epoch_loss = 0. # 初始化每轮的损失。
            epoch_loss_deavrg = 0. # 初始化每轮的平均损失。

            if dsettype == 'train': 
                for model, model_name in models:
                    model.train()
            else:
                for model, model_name in models:
                    model.eval()
            # 迭代数据加载器的批次。
            for i_batch, samples_batch in enumerate(data_loader):
                '''
                i_batch: 200
                    0
                    1
                    2
                samples_batch: 
                    X_feat: 200*20  y_score: 200  y_categ: 200  indx: 200  indx_seqid_map: 200
                    [[[2, 1, 1,  ..., 1, 2, 3], # X_feat 200*20
                      ...
                      [2, 0, 1,  ..., 0, 1, 2]],
                     [0.0101, 0.1950, ..., 0.0000, 0.0195], # y_score 1*200
                     [1, 1, ..., 0, 1], # y_categ 1*200
                     [5235,  3718, ..., 3100,  9161], # indx 1*200
                     ('RANDOMseqNO25749', 'RANDOMseqNO18321', ..., 'RANDOMseqNO15316', 'RANDOMseqNO45705') # indx_seqid_map: 1*200
                '''
                print('batch num:', i_batch)

                # zero model grad
                if dsettype == 'train':
                    optimizer.zero_grad()

                # X_batch, __ , y_batch, b_seqs_indx, b_seqs_id = samples_batch # X_batch 200*20 y_batch: 1*200
                X_batch, y_score , y_batch, b_seqs_indx, b_seqs_id = samples_batch # X_batch 200*20 y_batch: 1*200
                # print(y_batch.shape)
                '''
                X_batch:
                    [[2, 1, 1,  ..., 1, 2, 3], # X_feat 200*20
                      ...
                     [2, 0, 1,  ..., 0, 1, 2]]
                y_batch:
                    [1, 1, ..., 0, 1] # y_categ 1*200
                b_seqs_id:
                    ('RANDOMseqNO25749', 'RANDOMseqNO18321', ..., 'RANDOMseqNO15316', 'RANDOMseqNO45705') # indx_seqid_map: 1*200
                '''

                X_batch = X_batch.to(device) # 将输入数据移动到设备上。
                y_batch = y_batch.type(torch.int64).to(device) # 将标签数据移动到设备上。

                with torch.set_grad_enabled(dsettype == 'train'): # 根据当前数据集类型设置是否计算梯度。
                    num_samples_perbatch = X_batch.size(0) # 获取当前批次的样本数量。
                    # X_batch 200*20
                    logsoftmax_scores, fattn_w_scores, attn_mlayer_mhead_dict = criscas_categ_model(X_batch) # logsoftmax_scores 200*2 fattn_w_scores 2*64


                    if dsettype in seqid_fattnw_map: # 如果当前数据集类型在注意力权重字典中。
                        seqid_fattnw_map[dsettype].update({seqid:fattn_w_scores[c].detach().cpu() for c, seqid in enumerate(b_seqs_id)}) # 更新验证和测试集的注意力权重字典。
                        # seqid_mlhattnw_map[dsettype].update(process_multilayer_multihead_attn(attn_mlayer_mhead_dict, b_seqs_id))

                    __, y_pred_clss = torch.max(logsoftmax_scores, 1) # 获取预测的类别。 # 1*200 eg: [1, 1, 1, ..]
                    # print(y_pred_clss.shape) # 200*2

                    pred_class.extend(y_pred_clss.view(-1).tolist()) # 将预测的类别添加到预测类别列表中。
                    ref_class.extend(y_batch.view(-1).tolist()) # 将标签类别添加到参考类别列表中。
                    prob_scores.append((torch.exp(logsoftmax_scores.detach().cpu())).numpy()) # 将概率分数添加到概率分数列表中。
                    seqs_ids_lst.extend(list(b_seqs_id)) # 将序列ID添加到序列ID列表中。

                    loss = loss_func(logsoftmax_scores, y_batch) 

                    if dsettype == 'train': 
                        # print("computing loss")
                        # backward step (i.e. compute gradients)
                        loss.backward() # 反向传播计算梯度
                        # optimzer step -- update weights
                        optimizer.step() # 更新模型参数
                        # after each batch step the scheduler
                        cyc_scheduler.step() # 更新学习率调度器
                    epoch_loss += loss.item() # 累加每轮的损失
                    # deaverage the loss to deal with last batch with unequal size
                    epoch_loss_deavrg += loss.item() * num_samples_perbatch # 累加每轮的平均损失

                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader)) # 将每轮的批次平均损失添加到列表中。
            epoch_loss_avgsamples[dsettype].append(epoch_loss_deavrg/len(data_loader.dataset)) # 将每轮的样本平均损失添加到列表中。
            prob_scores_arr = np.concatenate(prob_scores, axis=0) # 将概率分数列表拼接为数组。
            modelscore = perfmetric_report_categ(pred_class, ref_class, prob_scores_arr[:, 1], epoch, flog_out[dsettype]) # 计算分类性能指标。
            perf = modelscore.auc # 获取AUC性能指标。
            if(perf > score_dict[dsettype].auc): # 如果当前性能指标大于之前记录的最佳性能指标。
                score_dict[dsettype] = modelscore # 更新最佳分类性能指标。
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        # 如果当前数据集类型为验证集, 保存模型的状态字典。
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                    # dump attention weights for the validation data for the best peforming model
                    # dump_dict_content(seqid_fattnw_map, ['validation'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['validation'], 'seqid_mlhattnw_map', wrk_dir)
                elif(dsettype == 'test'):
                    # dump attention weights for the validation data
                    # 将验证集或测试集的注意力权重字典保存到磁盘
                    dump_dict_content(seqid_fattnw_map, ['test'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['test'], 'seqid_mlhattnw_map', wrk_dir)
                    # save predictions for test
                if dsettype in {'test', 'validation'}:
                    # 构建预测结果的数据框。
                    predictions_df = build_predictions_df(seqs_ids_lst, ref_class, pred_class, prob_scores_arr)
                    # 构建保存预测结果的文件路径。
                    predictions_path = os.path.join(wrk_dir, f'predictions_{dsettype}.csv')
                    # 将预测结果保存为CSV文件。
                    predictions_df.to_csv(predictions_path)

    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, fig_dir) # 绘制损失曲线图。

    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir) #将分类性能指标字典保存到磁盘。


def build_predictions_df(ids, true_class, pred_class, prob_scores):
    '''构建预测结果的 DataFrame'''
    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class,
        'prob_score_class1': prob_scores[:,1],
        'prob_scores_class0': prob_scores[:,0]
    }
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def generate_hyperparam_space(model_name):
    '''生成超参数空间'''
    if(model_name == 'Transformer'):
        # TODO: add the possible options for transformer model
        embed_dim = [16,32,64,128]
        num_attn_heads = [4,6,8,12]
        num_transformer_units = [2,4,6,8]
        p_dropout = [0.1, 0.3, 0.5]
        nonlin_func = [nn.ReLU, nn.ELU]
        mlp_embed_factor = [2]
        l2_reg = [1e-4, 1e-3, 0.]
        batch_size = [200, 400, 600]
        num_epochs = [30]
        opt_lst = [embed_dim, num_attn_heads, 
                   num_transformer_units, p_dropout,
                   nonlin_func, mlp_embed_factor,
                   l2_reg, batch_size, num_epochs]

    hyperparam_space = list(itertools.product(*opt_lst))

    return hyperparam_space

def compute_numtrials(prob_interval_truemax, prob_estim):
    '''计算进行随机超参数搜索所需的迭代次数'''
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper
        <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
        Args:
            prob_interval_truemax: float, probability interval of the true optimal hyperparam,
                i.e. within 5% expressed as .05
            prob_estim: float, probability/confidence level, i.e. 95% expressed as .95
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)


def get_hyperparam_options(prob_interval_truemax, prob_estim, model_name, random_seed=42):
    '''获取随机超参数搜索的超参数选项'''
    np.random.seed(random_seed)
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    hyperparam_space = generate_hyperparam_space(model_name)
    if(num_trials > len(hyperparam_space)):
        num_trials = len(hyperparam_space)
    indxs = np.random.choice(len(hyperparam_space), size=num_trials, replace=False)
    if(model_name == 'Transformer'):
        hyperconfig_class = TrfHyperparamConfig
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    return [hyperconfig_class(*hyperparam_space[indx]) for indx in indxs]



def get_saved_config(config_dir):
    '''加载已保存的模型配置。'''
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_index_argmax(score_matrix, target_indx):
    '''获取得分矩阵中指定目标索引的最大值索引'''
    argmax_indx = np.argmax(score_matrix, axis=0)[target_indx]
    return argmax_indx


def train_val_run(datatensor_partitions, config_map, train_val_dir, run_gpu_map, num_epochs=20):
    '''运行训练和验证'''
    dsettypes = ['train', 'validation']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs # 50
    for run_num in datatensor_partitions: # 迭代每一份
        options['run_num'] = run_num # 0
        dataset_fold = datatensor_partitions[run_num] 
        path = os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num))
        wrk_dir = create_directory(path) # /Users/jarrycf/Desktop/BEtools2.0/perbase_train/ABEmax/train_val/run_0
        run_categTrf(dataset_fold, dsettypes, mconfig, options, wrk_dir,
                     state_dict_dir=None, to_gpu=True, gpu_index=run_gpu_map[run_num])


def test_run(datatensor_partitions, config_map, train_val_dir, test_dir, run_gpu_map, num_epochs=1):
    '''运行测试'''
    dsettypes = ['test']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs # 使用用户指定的值覆盖迭代次数
    for run_num in datatensor_partitions: 
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        train_dir = create_directory(os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num)))
        if os.path.exists(train_dir):
            # 加载静态路径
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, 'test', 'run_{}'.format(run_num))
            test_wrk_dir = create_directory(path)
            run_categTrf(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                         state_dict_dir=state_dict_pth, to_gpu=True, gpu_index=run_gpu_map[run_num])
        else:
            print('WARNING: train dir not found: {}'.format(path))
