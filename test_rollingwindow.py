import pandas as pd 
import numpy as np 

list = [i for i in range(100)]
df = pd.DataFrame(list)
df.columns = ['time']
# print(df)

# split_df = np.array_split(df,10)
# print(split_df[0])

def RollingWindow(df: pd.DataFrame): 

    start = 0 
    curr = 0 
    end = len(df) # 100 entries 
    slices = 20 
    increment = int(end/slices) # train window is 5
    desired_window_size = 10 # desired_window size is 10

    while curr < (end - increment):
        
        # train indexes 
        train_idx_front = start 
        train_idx_back = curr + increment
        curr_win_size = train_idx_back - train_idx_front 

        # test indexes
        test_idx_front = train_idx_back 
        test_idx_back = test_idx_front + increment 

        # Obtain train and test data 
        train_df = df.iloc[train_idx_front:train_idx_back] 
        test_df = df.iloc[test_idx_front:test_idx_back]
        
        # increment window until desired size is reached
        if curr_win_size == desired_window_size: 

            # move start pointer to curr pointer's position
            start = curr 

        # increment curr pointer 
        curr += increment 
    
    

RollingWindow(df)





