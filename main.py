from config import opt
from data_handler import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import ImgModule, TxtModule
from utils import calc_map_k
import matplotlib.pyplot as plt

#将print结果保存至txt
import sys

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger('./result.txt')  # 调用print时相当于Logger().write()


def train(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = tags.shape[1]
    l_dim = labels.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit, pretrain_model)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])
    #注意！label标签并不是浮点数类型，下面运算会出错，所以开始进行转换方便下面计算
    train_L = train_L.type(torch.float)

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    num_train = train_x.shape[0]

    #初始化各矩阵
    #用var{}设置为全局变量会更加保证正确性
    #var{"W1"} = torch.randn(l_dim,opt.bit)
    W1 = torch.randn(l_dim,opt.bit)
    W2 = torch.randn(l_dim,opt.bit)
    #l-n*24;f-n*64故W-24*64
    I = torch.ones(opt.bit,opt.bit)
    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)
    #train_L的运算备用
    LL = Variable(train_L)
    if opt.use_gpu:
        W1 = W1.cuda()
        W2 = W2.cuda()
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()
        I = I.cuda()
        LL = LL.cuda()

    #相似矩阵
    Sim = calc_neighbor(train_L, train_L)
    #哈希编码
    B = torch.sign(F_buffer + G_buffer)
    #训练规模
    batch_size = opt.batch_size
    #学习率（步长）linspace设置等差数列
    lr = opt.lr
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)
    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }
    loss_all = {
        'loss':[]
    }
    
    loss_list = []
    loss_y_list = []
    loss_x_list = []
    map_list = []
    mapi2t_list = []
    mapt2i_list = []
    loss_x_sum = 0.
    loss_y_sum = 0.
    #全一要用
    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)
    unupdated_size = num_train - batch_size
    #两种map
    max_mapi2t = max_mapt2i = 0.

    for epoch in range(opt.max_epoch):
        loss_x_sum = 0.
        loss_y_sum = 0.        
        # 训练图像网络
        for i in tqdm(range(num_train // batch_size)):
            torch.autograd.set_detect_anomaly(True)
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()
                ones_ = ones_.cuda()

            # similar matrix size--S: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)
            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)
                       
            #反向传播计算损失，要用.t()或者.transpose()转置
            # theta_x: (batch_size, num_train)
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))            
            labeldis_x = torch.matmul(cur_f,W1.t())-sample_L
            labelloss_x = torch.sum(torch.pow(labeldis_x,2))
            balance_w1 = torch.sum(torch.pow(W1,2))
            # loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x + labelloss_x +balance_w1
            loss_x = logloss_x * opt.aph1 + opt.gamma * quantization_x + opt.eta * balance_x + balance_w1 * opt.lamda + labelloss_x * opt.aph2
            loss_x = loss_x /(batch_size * num_train)

            optimizer_img.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss_x.backward()
            optimizer_img.step()
            loss_x_sum = loss_x_sum+loss_x

        # 训练文本网络
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)
            cur_g = txt_model(text)  # cur_g: (batch_size, bit)
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            # theta_y: (batch_size, num_train)
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            
            labeldis_y = torch.matmul(cur_g,W2.t())-sample_L
            labelloss_y = torch.sum(torch.pow(labeldis_y,2))
            balance_w2 = torch.sum(torch.pow(W2,2))
            
            loss_y = logloss_y * opt.aph1 + opt.gamma * quantization_y + opt.eta * balance_y + labelloss_y * opt.lamda + balance_w2 * opt.aph2
            loss_y = loss_y / (num_train * batch_size)

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            loss_y_sum = loss_y_sum+loss_y

        # update B
        B = torch.sign(F_buffer + G_buffer)
        ### update W1 & W2     
        #偏导W -2LtF+2WFtF+2λW
        #另其=0优化W得 W=LtF(FtF+λI)^-1

        q1=torch.matmul(LL.t(),F_buffer)
        q2=torch.inverse(torch.matmul(F_buffer.t(),F_buffer)*opt.aph2 + I * opt.lamda)
        W1=torch.matmul(q1*opt.aph2,q2)
        p1=torch.matmul(LL.t(),G_buffer)
        p2=torch.inverse(torch.matmul(G_buffer.t(),G_buffer)*opt.aph2 + I * opt.lamda)
        W2=torch.matmul(p1*opt.aph2,p2)
        # W1 = torch.matmul(torch.matmul(train_L.transpose(0,1),F),torch.inverse(torch.matmul(F.transpose(0,1),F)+I))
        # W2 = torch.matmul(torch.matmul(train_L.transpose(0,1),G),torch.inverse(torch.matmul(G.transpose(0,1),G)+I))
        
        # calculate total loss
        loss = calc_loss(B, F, G, W1, W2, train_L, Variable(Sim), opt.gamma, opt.eta, opt.lamda, opt.aph1 , opt.aph2)
        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
        result['loss'].append(float(loss.data))
        loss_list.append(loss)
        loss_x_list.append(loss_x_sum/(num_train // batch_size))
        loss_y_list.append(loss_y_sum/(num_train // batch_size))

        if opt.valid:
            mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                   query_L, retrieval_L)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                img_model.save(img_model.module_name + '.pth')
                txt_model.save(txt_model.module_name + '.pth')

        lr = learning_rate[epoch + 1]

        # 设置学习步长
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

    x1 = range(0, opt.max_epoch)
    y1 = loss_list
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, '.-')
    plt.title('loss vs. epoches')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig("loss.jpg")
    
    x2 = range(0, opt.max_epoch)
    y2 = loss_x_list
    plt.subplot(2, 2, 2)
    plt.plot(x2, y2, '.-')
    plt.title('loss_x vs. epoches')
    plt.xlabel('epoches')
    plt.ylabel('loss_x')
    # plt.show()
    plt.savefig("loss_x.jpg")

    x3 = range(0, opt.max_epoch)
    y3 = mapt2i_list
    plt.subplot(2, 2, 3)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epoches')
    plt.ylabel('mapt2i')
    # plt.show()
    plt.savefig("mapti2.jpg")

    x4 = range(0, opt.max_epoch)
    y4 = mapi2t_list
    plt.subplot(2, 2, 4)
    plt.plot(x4, y4, '.-')
    plt.xlabel('epoches')
    plt.ylabel('mapi2t')
    # plt.show()
    plt.savefig("./result/mapi2t.jpg")

    # x3 = range(0, opt.max_epoch)
    # y3 = loss_y_list
    # plt.subplot(2, 2, 3)
    # plt.plot(x3, y3, '.-')
    # plt.title('loss_y vs. epoches')
    # plt.xlabel('epoches')
    # plt.ylabel('loss_y')
    # # plt.show()
    # plt.savefig("loss_y.jpg")

    loss_all['loss'] = loss_list
    loss_all['loss_x'] = loss_x_list
    loss_all['loss_y'] = loss_y_list
    map_list['i2t']=mapi2t_list
    map_list['t2i']=mapt2i_list

    write_result(result)
    write_loss(loss_all)
    write_map(map_list)


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L

def calc_neighbor(label1, label2):
    # 计算相似矩阵
    if opt.use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


    #l-128*24;f-128*64;W-24*64
    #L-FWt  Lt-WFt
    #偏导F -2LW+2FWtW
    #偏导W -2LtF+2WFtF+2λW
    #另其=0优化W得 W=LtF(FtF+λI)^-1    
    #利用F.sum(dim=0)纵向压缩，将每列的数求和，变成一维向量
def calc_loss(B, F, G, W1, W2, L, Sim, gamma, eta, lamda, aph1 , aph2):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term3 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term4 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    term21 = torch.sum(torch.pow(torch.matmul(F,W1.transpose(0,1))-L,2))*aph2+torch.sum(torch.pow(W1,2))*lamda
    term22 = torch.sum(torch.pow(torch.matmul(F,W2.transpose(0,1))-L,2))*aph2+torch.sum(torch.pow(W2,2))*lamda
    term2 = term21 + term22

    loss = aph1*term1 + gamma * term3 + eta * term4 + term2*aph2
    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B

def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result):
    import os
    with open(os.path.join(opt.result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')

def write_loss(loss):
    import os
    with open(os.path.join(opt.result_dir, 'loss.txt'), 'w') as f:
        for k, v in loss.items():
            f.write(k + ' ' + str(v) + '\n')

def write_map(map):
    import os
    with open(os.path.join(opt.result_dir, 'map.txt'), 'w') as f:
        for k, v in map.items():
            f.write(k + ' ' + str(v) + '\n')

def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()
    # test()
