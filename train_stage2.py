import torch
import argparse
import os
import net
import TangkaNet
import config
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# import Imageset
import Imageset2
import torch.nn.functional as F

import sys
sys.argv=['']
del sys

# train stage two - fine-tune
def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))   #config.gpu_name
       
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset = Imageset2.get_train_data()
    eval_dataset = Imageset2.get_test_data()
    train_data = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)
    eval_data = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)
    model = TangkaNet.SimCLRStage2(num_class=12).to(DEVICE)   #这段代码的意思就是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    model.load_state_dict(torch.load(args.pre_model),strict=False)
    ############loss###########
    #loss1
    loss_criterion = TangkaNet.GHMC()
    
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=config.stage2_lr, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch+1):
        model.train()
        total_loss = 0
        for batch, (data, target) in enumerate(train_data):
        
            target = torch.as_tensor(target)##
            data_device, target = data.to(DEVICE), target.to(DEVICE)
        
            
            pred = model(data_device,data)
            loss = loss_criterion(pred,target,epoch,batch)
            
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch", epoch, "loss:", total_loss / len(train_dataset) * args.batch_size)
        with open(os.path.join(config.save_path, "stage2_loss_train.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * args.batch_size) + '\n')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(eval_data):
                    data_device, target = data.to(DEVICE), target.to(DEVICE)
                    pred = model(data_device,data)

                    total_num += data.size(0)
                    print("data.size:")
                    print(data.size(0))
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc
                    total_loss += loss.item()
                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))
                print("total num:")
                print(total_num)
                print("eval_data:")
                print(len(eval_data))
                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                          "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))
                with open(os.path.join(config.save_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + '\n')
                with open(os.path.join(config.save_path, "stage2_top5_acc.txt"), "a") as f:
                    f.write(str(total_correct_5 / total_num * 100) + '\n')
                with open(os.path.join(config.save_path, "stage2_loss_test.txt"), "a") as f:
                    f.write(str(total_loss / total_num * args.batch_size) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=config.stage2_batch_size, type=int, help='')
    parser.add_argument('--max_epoch', default=config.stage2_max_epoch, type=int, help='')
    parser.add_argument('--pre_model', default=config.pre_model, type=str, help='')

    args = parser.parse_args()
    train(args)
