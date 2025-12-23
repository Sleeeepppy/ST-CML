import argparse
import torch
import yaml
import time
from tqdm import tqdm
from datasets import traffic_dataset
from torch_geometric.data import DataLoader
from maml import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device name')
parser.add_argument('--seed',type=int, default=0, help='random seed')
parser.add_argument('--model', default='GRU', type=str)
# contrastive:64,meta:8
parser.add_argument('--source_epochs', type=int, default=120, help='epochs')
parser.add_argument('--target_epochs', type=int, default=20, help='epochs')
parser.add_argument('--target_days', type=int, default=3, help='few shot training days')
parser.add_argument('--test_dataset', default='metr-la', type=str, help='test dataset')
parser.add_argument('--model_config_filename', default='./config/model_config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--data_config_filename', default='./config/data_config.yaml', type=str,
                        help='Configuration filename for data.')
parser.add_argument('--memo', default='revise', type=str)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.data_config_filename) as f:
        data_config = yaml.safe_load(f)
    with open(args.model_config_filename) as f:
        model_config = yaml.safe_load(f)

    # set random seed
    torch.manual_seed(args.seed)

    data_args, task_args, con_args, model_args = (data_config['data'], model_config['task'],
                                                  model_config['contrastive'], model_config['model'])
    print(model_args, '\n', task_args, '\n', con_args, '\n', data_args)

    # define data
    source_dataset = traffic_dataset(data_args, task_args, con_args, "source", args, True,
                                     test_data=args.test_dataset, target_days=args.target_days)
    model = STMAML(data_args, task_args, con_args, model_args, model=args.model).to(device=args.device)
    nparam = sum([p.nelement() for p in model.parameters()])
    print('Total parameters:', nparam)
    # choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['update_lr'])
    # choose loss function
    loss_criterion = nn.MSELoss()

    source_training_losses, target_training_losses = [], []
    best_result = ''
    min_MAE = 10000000

    for epoch in tqdm(range(args.source_epochs)):
        # Meta-Train
        start_time = time.time()

        spt_task_data, spt_task_A, qry_task_data, qry_task_A, aug_spt_task_data, aug_spt_task_A, aug_qry_task_data, aug_qry_task_A \
            = source_dataset.get_maml_task_batch_aug(task_args['task_num'])

        # loss = model.meta_train_revise(spt_task_data, spt_task_A, qry_task_data, qry_task_A)

        loss = model.contrastive_meta_train(spt_task_data, spt_task_A, qry_task_data, qry_task_A,
                                            aug_spt_task_data, aug_spt_task_A, aug_qry_task_data, aug_qry_task_A)
        print(loss)
        # loss = model.meta_train(spt_task_data, spt_task_A, qry_task_data, qry_task_A)
        end_time = time.time()
        if epoch % 20 == 0:
            print("[Source Train] epoch #{}/{}: loss is {}, training time is {}".format(
                epoch+1, args.source_epochs, loss, end_time-start_time))

    print("Source dataset meta-train finish.")

    target_dataset = traffic_dataset(data_args, task_args, con_args,"target",args ,
                                     test_data=args.test_dataset, target_days=args.target_days)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['batch_size'],
                                   shuffle=True, num_workers=8, pin_memory=True)



    test_dataset = traffic_dataset(data_args, task_args, con_args,"test", args,test_data=args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'],
                                 shuffle=True, num_workers=8, pin_memory=True)

    model.finetuning(target_dataloader, test_dataloader, args.target_epochs, args.test_dataset)
    print(args.memo)