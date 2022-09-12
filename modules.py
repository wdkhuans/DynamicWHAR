import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWHAR (nn.Module):

    def __init__(self, node_num = 5, node_dim = 9, window_size=24, channel_dim=8, time_reduce_size=10, hid_dim=128, class_num=17):    
        super(DynamicWHAR, self).__init__()
        self.node_num = node_num
        self.node_dim = node_dim
        self.window_size = window_size
        self.channel_dim = channel_dim
        self.time_reduce_size = time_reduce_size
        self.hid_dim = hid_dim
        self.class_num = class_num
        
        self.dropout_prob = 0.6
        self.conv1 = nn.Conv1d(self.node_dim, self.channel_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(self.channel_dim)
        self.conv2 = nn.Conv1d(self.window_size, self.time_reduce_size, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(self.time_reduce_size)
        self.conv3 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, self.channel_dim * self.time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)        
        self.conv5 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, self.channel_dim * self.time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)
        
        self.msg_fc1 = nn.Linear(self.channel_dim * self.time_reduce_size * 3 * self.node_num, self.hid_dim)   
        self.fc_out  = nn.Linear(self.hid_dim, self.class_num)
        
        self.conv4 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, 1, kernel_size=1, stride=1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def edge2node(self, x, rel_rec, rel_send, rel_type):
        mask = rel_type.squeeze()
        x = x + x * (mask.unsqueeze(2))
        rel = rel_rec.t() + rel_send.t()
        incoming = torch.matmul(rel, x)
        return incoming / incoming.size(1)

    def forward(self, inputs, rel_rec, rel_send):
        x = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3])
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(inputs.shape[0], inputs.shape[1], x.shape[1])
        s_input_1 = x  
        
        edge = self.node2edge(s_input_1, rel_rec, rel_send)
        edge = edge.permute(0, 2, 1)       
        edge = F.relu(self.bn3(self.conv3(edge)))
        edge = edge.permute(0, 2, 1)
        
        x = edge.permute(0, 2, 1)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        rel_type = F.sigmoid(x)

        s_input_2 = self.edge2node(edge, rel_rec, rel_send, rel_type)
        s_input_2 = s_input_2.permute(0, 2, 1)
        s_input_2 = F.relu(self.bn5(self.conv5(s_input_2)))
        s_input_2 = s_input_2.permute(0, 2, 1)

        join = torch.cat((s_input_1, s_input_2), dim=2)
        join = join.reshape(join.shape[0], -1)
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        join = F.relu(self.msg_fc1(join))
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        preds = self.fc_out(join)        
        
        return preds