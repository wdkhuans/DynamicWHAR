import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import data

def edge_init(node_num, cuda):
    
    off_diag = np.ones([node_num, node_num]) - np.eye(node_num) # (5, 5)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    relation_num = node_num - 1
    rel_rec_undirected = np.empty([0, node_num])
    rel_send_undirected = np.empty([0, node_num])
    for k in range(1, relation_num + 1):
        rel_rec_undirected  = np.concatenate( (rel_rec_undirected,rel_rec[((k-1)*relation_num+k-1):(k*relation_num),:] ), axis=0)
        rel_send_undirected  = np.concatenate( (rel_send_undirected,rel_send[((k-1)*relation_num+k-1):(k*relation_num),:] ), axis=0)

    rel_rec_undirected = torch.FloatTensor(rel_rec_undirected)
    rel_send_undirected = torch.FloatTensor(rel_send_undirected)   

    if cuda:
        rel_rec_undirected = rel_rec_undirected.cuda()
        rel_send_undirected = rel_send_undirected.cuda()
        
    rel_rec_undirected = Variable(rel_rec_undirected)
    rel_send_undirected = Variable(rel_send_undirected)

    return rel_rec_undirected, rel_send_undirected

def load_data(name='opp_24_12', batch_size=64, test_user=0):
    
    if 'opp' in name:
        _dataset = data.Opportunity(name = name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
    
        invalid_feature = np.arange(0, 36)
        train_x  = np.delete( train_x, invalid_feature, axis = 2 )
        test_x  = np.delete( test_x, invalid_feature, axis = 2 )
    
        train_x = train_x[:,:,:45]
        test_x = test_x[:,:,:45]
    
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 9)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 9)
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])

    elif 'realdisp' in name:
        _dataset = data.Realdisp(name = name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
        
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 13)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 13)
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])
        
        train_x = train_x[:, :, :, :9]
        test_x = test_x[:, :, :, :9]
        
    elif 'realworld' in name:
        _dataset = data.Realworld(name = name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
        
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 9)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 9)
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])

    elif 'skoda' in name:
        _dataset = data.Skoda(name = name)
        train_x, train_y, val_x, val_y, test_x, test_y = _dataset.load_data(test_user)

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 3)
        val_x   = val_x.reshape(val_x.shape[0], val_x.shape[1], -1, 3)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 3)

        train_x = np.transpose(train_x, [0, 2, 1, 3])
        val_x = np.transpose(val_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])

    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    test_x  = torch.FloatTensor(test_x)
    test_y  = torch.LongTensor(test_y)  
    
    train_data = TensorDataset(train_x, train_y)
    test_data = TensorDataset(test_x, test_y)    

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader  = DataLoader(test_data, batch_size=batch_size)           
    
    return train_data_loader, test_data_loader

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot