from interface import Interface

if __name__=='__main__':
    Interface().train(epoch=30, learning_rate=1e-4, batch_size=36, save_freq=2, train_mode='Alternation')
    Interface().train(epoch=30, learning_rate=1e-4, batch_size=36, save_freq=2, train_mode='PBranch_Enc')
