from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

#데이터 로드
from datasets import load_dataset

dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

#데이터 전처리
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 256

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

train_encodings = train_data.map(preprocess_function, batched=True)
test_encodings = test_data.map(preprocess_function, batched=True)

#파인튜닝을 위한 모델 정의(긍정 부정 분류 태스크 학습)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 모델 학습을 위한 하이퍼파라미터와 옵티마이저 정의
batch_size = 16
num_epochs = 3
learning_rate = 2e-5

train_dataset = train_encodings.remove_columns(['text']).rename_column('label', 'labels')
test_dataset = test_encodings.remove_columns(['text']).rename_column('label', 'labels')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
``

#파인튜닝 과정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
        
        loss = criterion(logits, labels)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    
    print('Epoch {} - Loss: {:.4f} - Accuracy: {:.4f}'.format(epoch+1, epoch_loss, epoch_accuracy))

# 학습된 모델을 평가
model.eval()

total_predictions = 0
correct_predictions = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

accuracy = correct_predictions / total_predictions
print('Test Accuracy: {:.4f}'.format(accuracy))

##출력되는 점수가 높을 수록 좋은 평가를 받음.