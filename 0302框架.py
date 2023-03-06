import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Lambda, dot, Activation, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint

# 设置超参数
T = 4 # 时间步数
img_rows, img_cols = 64, 64 # 图像大小
input_shape = (img_rows, img_cols, 3) # 输入形状
num_classes = 6 # 类别数

# 定义CNN模型
def cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 定义LSTM模型
def lstm_model(T, input_shape, num_classes):
    inputs = Input(shape=(T,) + input_shape)
    x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 定义时间注意力机制模型
def attention_model(T, lstm_hidden_size):
    inputs = Input(shape=(T, lstm_hidden_size))
    attention_probs = Dense(T, activation='softmax', name='attention_probs')(inputs)
    attention_mul = Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes=1), name='attention_mul')([attention_probs, inputs])
    outputs = attention_mul
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 定义训练函数
def
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        # forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch+1, num_epochs, i+1, total_step, running_loss / 100))
            running_loss = 0.0
def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for i, data in enumerate(test_loader):1单位
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            
        accuracy = total_correct / total_samples
        loss = total_loss / (i+1)
        print('Test Accuracy: %.2f%%, Test Loss: %.4f' % (accuracy*100, loss))
        return accuracy, loss
def test(model, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 执行前向传递
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算分类准确度
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    # 计算平均损失和分类准确度
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = total_correct / total_samples

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2%}'.format(avg_loss, accuracy))
