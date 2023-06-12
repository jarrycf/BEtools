# win_left = eval(input("输入需要编辑的窗口范围"))

# win_right = eval(input("输入需要编辑的窗口范围"))
# win_left, win_right = map(int, input("输入需要编辑的窗口范围，用空格隔开：").split())
# print(win_left)
# import pickle

# with open('/Users/jarrycf/Desktop/BEtools2.0/trained_models/perbase/ABE8e/train_val/run_0/config/exp_options.pkl', 'rb') as f:

#     exp_options = pickle.load(f)

# # 查看字典的键
# keys = exp_options.keys()
# '''
# dict_keys(['experiment_desc', 'run_num', 'model_name', 'num_epochs', 'weight_decay', 'fdtype', 'to_gpu', 'loss_func', 'per_base', 'train_flag'])
# '''

# import torch

# # 加载模型状态字典并将其映射到CPU
# state_dict = torch.load('/Users/jarrycf/Desktop/BEtools2.0/trained_models/perbase/ABE8e/train_val/run_0/model_statedict/Transformer.pkl', map_location=torch.device('cpu'))

# # 查看键
# keys = state_dict.keys()
# print(keys)

