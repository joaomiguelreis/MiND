import numpy as np

class Label_Gen():
    '''
    Args:
        label_col (int, >4, <181): column to be considered the label
    '''
    # Make sure the argument is correct


    def __init__(self,  PT_data_complete):
        super(Label_Gen, self).__init__()
        self.PT_data_complete = PT_data_complete

    def __getitem__(self, label_col):
        assert 4<label_col<181
        
        # Create the new dataframe
        PT_data = self.PT_data_complete.copy()
        
        # Create the column with label
        col_name = PT_data.columns[label_col]
        PT_data['label'] = PT_data[col_name]
        
        # Delete all not needed columns
        PT_data = PT_data.drop(columns = PT_data.columns[5:-1])
        
        # Format the data for skorch use
        PT_numpy = PT_data.values
        X = PT_numpy[:,:-1]
        y = PT_numpy[:,-1]
        X = X.astype(np.float32)
        y = y.astype(np.int64)   
        return X,y

def mult_label_gen(PT_data):
    ices = np.zeros(175*1000)
    for i in range(len(PT_data)):
        ice = PT_data.loc[i].iloc[6:len(PT_data.columns)]
        ices[i*175:(i+1)*175] = ice.values

    PT_data = PT_data.iloc[np.repeat(np.arange(len(PT_data)), 175)]
    drop_icol = list(range(5,len(PT_data.columns)))

    PT_data = PT_data.drop(PT_data.columns[drop_icol],axis=1)
    PT_data['loc'] = np.array(list(range(175))*1000)
    PT_data['ice'] = ices
    PT_numpy = PT_data.values
    X = PT_numpy[:,:-1]
    y = PT_numpy[:,-1]
    X = X.astype(np.float32)
    y = y.astype(np.int64)   
    return X,y

