from interface import Interface

if __name__=='__main__':
    Interface().train(epoch=30, learning_rate=5e-5, batch_size=10, save_freq=2, train_mode='Alternation')
    Interface().train(epoch=30, learning_rate=4e-5, batch_size=10, save_freq=2, train_mode='PBranch_Enc')
