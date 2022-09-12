import numpy as np
from scipy.fftpack import fft
from sklearn import utils as skutils

class Opportunity( object ):

    def __init__( self, name = 'opp_24_12'):
        if name == 'opp_24_12':
            self._length = 24
        elif name == 'opp_60_30':
            self._length = 60             
            
        self.name               = name
        self._path              = 'Dataset/opp/'
        self._channel_num       = 113
        self._user_num          = 4
        self._act_num           = 17

    def load_data( self, test_user=0 ):

        train_x = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        train_y = np.empty( [0], dtype=np.int )

        test_x  = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        test_y  = np.empty( [0], dtype=np.int )

        for user_idx in range( self._user_num ):
            if user_idx == test_user: # training user
                test_x  = np.concatenate( (test_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                test_y  = np.concatenate( (test_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )
                
            else:
                train_x  = np.concatenate( (train_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                train_y  = np.concatenate( (train_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

        train_x, train_y    = skutils.shuffle( train_x, train_y )
        test_x, test_y      = skutils.shuffle( test_x, test_y )

        return train_x, train_y, test_x, test_y 
    
class Realworld( object ):

    def __init__( self, name = 'realworld_40_20'):
        if name == 'realworld_40_20':
            self._length = 40   
        elif name == 'realworld_100_50':
            self._length = 100  
            
        self.name               = name
        self._path              = 'Dataset/realworld/'
        self._channel_num       = 63
        self._user_num          = 13
        self._act_num           = 8    

    def load_data( self, test_user=0 ):

        train_x = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        train_y = np.empty( [0], dtype=np.int )

        test_x  = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        test_y  = np.empty( [0], dtype=np.int )

        for user_idx in range( self._user_num ):
            if user_idx == test_user: # training user
                test_x  = np.concatenate( (test_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                test_y  = np.concatenate( (test_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

            else:
                train_x  = np.concatenate( (train_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                train_y  = np.concatenate( (train_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

        train_x, train_y    = skutils.shuffle( train_x, train_y )
        test_x, test_y      = skutils.shuffle( test_x, test_y )

        return train_x, train_y, test_x, test_y

class Realdisp( object ):

    def __init__( self, name = 'realdisp_24_12'):
        if name == 'realdisp_40_20':
            self._length = 40
        elif name == 'realdisp_100_50':
            self._length = 100        
            
        self.name               = name
        self._path              = 'Dataset/realdisp/'
        self._channel_num       = 117
        self._user_num          = 10
        self._act_num           = 33        

    def load_data( self, test_user=0 ):

        train_x = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        train_y = np.empty( [0], dtype=np.int )

        test_x  = np.empty( [0, self._length, self._channel_num], dtype=np.float )
        test_y  = np.empty( [0], dtype=np.int )

        for user_idx in range( self._user_num ):
            if user_idx == test_user: # training user
                test_x  = np.concatenate( (test_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                test_y  = np.concatenate( (test_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

            else:
                train_x  = np.concatenate( (train_x, np.load(self._path+ self.name + '/sub{}_features.npy'.format(user_idx)) ), axis=0 )
                train_y  = np.concatenate( (train_y, np.load(self._path+ self.name + '/sub{}_labels.npy'.format(user_idx)) ), axis=0 )

        train_x, train_y    = skutils.shuffle( train_x, train_y )
        test_x, test_y      = skutils.shuffle( test_x, test_y )

        return train_x, train_y, test_x, test_y 

class Skoda( object ):

    def __init__( self, name = 'skoda_24_12'):
        if name == 'skoda_right_78_39':
            self._length = 78    
        elif name == 'skoda_right_196_98':
            self._length = 196
            
        self.name               = name
        self._path              = 'Dataset/skoda/'
        self._channel_num       = 30
        self._user_num          = 1
        self._act_num           = 10

    def load_data( self, test_user=-1 ):
        train_x = np.load(self._path+ self.name + '/train_data.npy')
        val_x   = np.load(self._path+ self.name + '/val_data.npy')
        test_x  = np.load(self._path+ self.name + '/test_data.npy')

        train_y = np.load(self._path+ self.name + '/train_label.npy')
        val_y   = np.load(self._path+ self.name + '/val_label.npy')
        test_y  = np.load(self._path+ self.name + '/test_label.npy')

        train_x, train_y    = skutils.shuffle( train_x, train_y )
        val_x, val_y        = skutils.shuffle( val_x, val_y )
        test_x, test_y      = skutils.shuffle( test_x, test_y )

        return train_x, train_y, val_x, val_y, test_x, test_y  