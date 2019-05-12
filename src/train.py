import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import os,sys,time,math,numpy,argparse,json
import matplotlib.pyplot as plt
import openpyxl as excel

from label import Dataset
from CNN import DsCNN as ConvNet

parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('-n','--name', default='None', type=str, help='the name of the model')
parser.add_argument('--cuda', default=0, type=int, help='number of cuda')
args = parser.parse_args()
netname = args.name
cuda_num=args.cuda
print('add pars successfully')

def main():
    
    configfile =    open("../conf/config.conf")
    jsonconfig =    json.load(configfile)
    info =          jsonconfig[netname]
    pre_loading=    info['pre_loading']
    batch_size=     info['batch_size']
    learning_rate=  info['learning_rate']
    qp=             info['qp']
    depth=          info['depth']
    patch_size=     info['patch_size']
    ReduceLR=       info['ReduceLR']
    weight_decay=   info['weight_decay']
    losskind=       info['losskind']
    num_epochs=     info['num_epochs']
    test_name=      info['test_dataset']
    train_name=     info['train_dataset']
    learning_rate=  float(learning_rate)
    psnr_max=       info['psnr_max']
    unitnum=        info['unitnum']
    print('load conf successfully')
    configfile.close()


    # Device configuration
    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() else 'cpu')
    #dataset
    train_dataset = Dataset(os.path.join(train_name,'img/{}'.format(qp)),os.path.join(train_name,'lab'),train=True,patch_size=patch_size)
    #Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # test Data loader                                           
    test_dataset = Dataset(os.path.join(test_name,'img/{}'.format(qp)),os.path.join(test_name,'lab'),train=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                                shuffle=False)
    #create or load model
    
    model = ConvNet(depth,unitnum)
    if os.path.exists(pre_loading):
        model.load_state_dict(torch.load(pre_loading))
        print('load model successfully')
    
    # Loss and optimizer
    if losskind =='L1':
        criterion = nn.L1Loss()
    else:
        if losskind =='MSE':
            criterion = nn.MSELoss()
        else:
            print('not exist such loss function\n')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay,momentum=0.9)
    if ReduceLR is True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8, patience=8000, verbose=True)
    print('set loss&opt successfully')
    model.to(device) 
    print('model to {} successfully'.format(device))
    #record some params
    ##caculate average loss
    loss_sum=0
    num_sum=0
    ##save loss_min & psnr_max
    loss_min=1
    ##save each loss&psnr value
    loss_plt=[]
    psnr_plt=[]
    print('start training:')
    for epoch in range(num_epochs):
        #cac average loss for each epoch
        loss_sum=0
        num_sum=0
        #when( avg loss<loss_min),change flag to cac PSNR
        save_flag=0
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.type(torch.cuda.FloatTensor).to(device)
            labels = labels.type(torch.cuda.FloatTensor).to(device)
            # Forward pass
            '''
            outputs = model(images).type(torch.cuda.FloatTensor)
            loss = criterion(outputs.type(torch.FloatTensor), labels.type(torch.FloatTensor))
            '''
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ReduceLR is True:
                scheduler.step(loss)
            # caculate loss
            loss_sum+=loss.item()
            num_sum+=1
            
        print('Epoch [{}/{}], Loss: {:.8f} , min: {}, psnr_max: {}'.format(epoch + 1, num_epochs, loss_sum/num_sum,loss_min,psnr_max))
        loss_plt.append(loss_sum/num_sum)
        #save model
        if loss_min>(loss_sum/num_sum):
            loss_min=loss_sum/num_sum
            if epoch>(num_epochs/10):
                #torch.save(model.state_dict(), netname)
                save_flag=1

        #evaluate
        if (epoch+1)%(num_epochs/50)==0 or save_flag==1:
            print('evaluating PSNR ...')
            with torch.no_grad():
                model.eval()
                num_eva=0
                sum_eva=0
                for images1, labels1 in test_loader:
                    images1  = images1.type(torch.cuda.FloatTensor).to(device)
                    labels1 = labels1.type(torch.cuda.FloatTensor).to(device)
                    outputs1 = model(images1)
                    psnr_value=calc_psnr(outputs1,labels1,0,1.0)
                    psnr_value_img=calc_psnr(images1,labels1,0,1.0)
                    sum_eva +=psnr_value-psnr_value_img
                    num_eva +=1 
                psnr_plt.append(sum_eva/num_eva)
                print('\n   ----------\nPSNR_AVG : {}\n   ----------\n'.format(sum_eva/num_eva))
                print('saving psnr to txt')
                f=open("../results/{}_psnr_record.txt".format(netname.split('.')[0]), 'a+')
                f.write('\ntime:{} epoch: {} psnr:{}'.format(time.asctime(),epoch,sum_eva/num_eva))
                f.close()
                if sum_eva/num_eva>psnr_max:
                    psnr_max=sum_eva/num_eva
                    print('saving model...')
                    torch.save(model.state_dict(), os.path.join('../model/',netname))
    #save results
    #save excel results
    print('saving final results...')
    if os.path.exists('../results/record.xlsx'):
        wb = excel.load_workbook('../results/record.xlsx')
        ws=wb.active
    else:
        wb=excel.Workbook()
        ws=wb.active
    ws.append(['new result'])
    ws.append([time.asctime(),netname])
    ws.append(loss_plt)
    ws.append(psnr_plt)
    wb.save('../results/record.xlsx')
    #save txt results
    '''
    f=open("./results/psnr.txt", "a+")
    f.write("\n\n -----------------\ntime :{}\nnet_kind : {}\nlr : {}\nqp : {}\nmin_loss : {}\nnum_epochs : {}\nLOSS IMG : {}\nPSNR IMG : {}\n".format(time.asctime(),kind,learning_rate,qp,loss_min,num_epochs))
    f.write('patch_size : {}\nweight decay : {}\ndropout : {}\nloss : {}\n'.format(patch_size,weight_decay,dropout,losskind))
    f.write('\n****** P S N R ******')
    f.write('{}'.format(psnr_plt))
    f.close() 
    f=open("./results/loss.txt", "a+")
    f.write(" -----------------\ntime :{}\nnet_kind : {}\nlr : {}\nqp : {}\nmin_loss : {}\nnum_epochs : {}\nLOSS IMG : {}\nPSNR IMG : {}\n".format(time.asctime(),kind,learning_rate,qp,loss_min,num_epochs))
    f.write('patch_size : {}\nweight decay : {}\ndropout : {}\nloss : {}\n'.format(patch_size,weight_decay,dropout,losskind))
    f.close() 
    '''
    #save loss&psnr picture

    '''
    #save the result of training
    psnrIMG_name="./PSNR/psnr_{}_{}_{}_{}_{}_{}_{}_{}.png".format(kind,qp,learning_rate,time.asctime(),patch_size,weight_decay,dropout,losskind)
    lossIMG_name="./LOSS/loss_{}_{}_{}_{}_{}_{}_{}_{}.png".format(kind,qp,learning_rate,time.asctime(),patch_size,weight_decay,dropout,losskind)

    x=numpy.linspace(1, num_epochs, num_epochs)
    plt.plot(x,numpy.array(loss_plt) )
    plt.savefig(lossIMG_name)
    
    x1=numpy.linspace(1, num_epochs/50, num_epochs/50)
    plt.plot(x1,numpy.array(psnr_plt) )
    plt.savefig(psnrIMG_name)
    '''
def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    sr = sr.detach().cpu()
    hr = hr.detach().cpu()
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :,:, :]
    mse = valid.pow(2).mean()
    #if mse==0:
        #mse=100000
    #print(mse)
    return -10 * math.log10(mse)

if __name__ == '__main__':
    main()
