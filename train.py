# -*- coding: utf-8 -*-
import torch

def evaluate(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = samples_number = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            samples_number += labels.size(0)
            
        average_loss = total_loss/(idx+1)
        accuracy = correct/samples_number
    return average_loss, accuracy
        

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    total_loss = 0.0
    correct = samples_number = 0
    for idx, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        samples_number += labels.size(0)
        
    average_loss = total_loss/(idx+1)
    accuracy = correct/samples_number
    
    return average_loss, accuracy

    
def train_model(model, model_param, epochs, device, train_dataloader, val_dataloader, criterion, optimizer, logger, run_name):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        train_loss, train_acc = train_one_epoch(model, device, train_dataloader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, device, val_dataloader, criterion)
        logger.update_metrics(train_loss, val_loss, train_acc, val_acc)
        
        has_improved = logger.update_monitor_metric()
        if has_improved:
            checkpoint = {"model":model.state_dict(), "model_param":model_param}
            logger.save_checkpoint(checkpoint, epoch)
            
    # print final training result
    logger.print_training_result() 
    logger.save()
    print('Finish Training')