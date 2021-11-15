import torch
from utils import *
##Cross Validation
from sklearn.model_selection import KFold
import time


# main
def train(dataloader, model, loss,device,num_epochs, n_folds = 5):
    results = {}
    folds = KFold(n_splits = n_folds)
    
    # KFold Cross Validation
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(dataloader)):
        print("fold n°{}".format(fold_+1))
        ##Split by folder and load by dataLoader
        train_subsampler = torch.utils.data.SubsetRandomSampler(trn_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=16, sampler=train_subsampler)
        valid_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=16, sampler=valid_subsampler)


        # Initialize Model
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):        
            model.train()
            start_time = time.time()
            train_loss = []
            train_iou  = [] 
            valid_loss = []
            valid_iou  = []           
            for batch_idx, (features,targets) in enumerate(train_dataloader):
           
                features = features.to(device)
                targets  = targets.to(device)        
                optimizer.zero_grad()

                ### FORWARD AND BACK PROP
                logits = model(features)
                cost = loss(logits, targets)            
                cost.backward()
                iou = iou_score(targets,logits).item()*100

                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                ### LOGGING
                train_loss.append(cost.item())
                train_iou.append(iou)

                if batch_idx != 0 and not batch_idx % 30:
                    print ('Epoch: %03d/%03d | time: %5.1f s | Batch %03d/%03d | Train Loss: %.4f | Train IoU: %.4f%% '
                           %(epoch+1, num_epochs, time.time()-start_time, batch_idx,
                             len(train_dataloader),
                             np.mean(train_loss),
                             np.mean(train_iou))
                          )

            ##Valid
            model.eval()                
            with torch.no_grad():
                for batch_idx, (features,targets) in enumerate(valid_dataloader):

                    features = features.to(device)
                    targets  = targets.to(device)  

                    logits = model(features)
                    cost   = loss(logits, targets)
                    iou    = iou_score(targets,logits).item()*100

                    ### LOGGING
                    valid_loss.append(cost.item())
                    valid_iou.append(iou)
                print('Epoch: %03d/%03d |  Valid Loss: %.4f | Valid IoU: %.4f%%' % (
                      epoch+1, num_epochs, 
                      np.mean(valid_loss),
                      np.mean(valid_iou)))
        results[fold_+1] = np.mean(valid_iou)

    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

