import torch
import argparse
import os
import net
import config
import load_dataset
# import preData
from torch.utils.data import DataLoader

# train stage one - contrastive learning
def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset = load_dataset.PreDataset(root='cifar_dataset', train=True, transform=config.train_transform, download=True)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, drop_last=True)
  
        
    model = net.SimCLRStage1().to(DEVICE)
    lossLR = net.Loss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0
        for batch, (imgL, imgR, targ) in enumerate(train_data):#batch为第几个batch
            imgL, imgR = imgL.to(DEVICE), imgR.to(DEVICE)
           
            _, pre_L = model(imgL)
            _, pre_R = model(imgR)
            
            loss = lossLR(pre_L, pre_R, args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        print("epoch loss:", total_loss / len(train_dataset) * args.batch_size)

        with open(os.path.join(config.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * args.batch_size) + '\n')

        if epoch % 100 == 0:
            for param in model.state_dict():
                print(param,'\t',model.state_dict()[param].size())
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser(description='Train SimCLR')  #创建一个ArgumentParser类的对象
    parser.add_argument('--batch_size', default=config.stage1_batch_size, type=int, help='') #add_argument
    parser.add_argument('--max_epoch', default=config.stage1_max_epoch, type=int, help='')
    parser.add_argument('--lr', default=config.stage1_lr, type=float, help='')

    args = parser.parse_args() 
    train(args)
