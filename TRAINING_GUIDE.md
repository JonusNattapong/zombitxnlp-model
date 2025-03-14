# คู่มือการฝึกสอนโมเดล ZombitX NLP

คู่มือนี้จะอธิบายขั้นตอนการฝึกสอนโมเดลประมวลผลภาษาไทยโดยใช้กรอบการทำงาน ZombitX NLP ซึ่งรองรับงานประมวลผลภาษาธรรมชาติหลากหลายรูปแบบ

## ขั้นตอนการฝึกสอนโมเดล

### 1. การเตรียมข้อมูล

การเตรียมข้อมูลเป็นขั้นตอนสำคัญในการฝึกสอนโมเดล โดยทั่วไปจะประกอบด้วยการทำความสะอาดข้อมูล การตัดคำ และการแบ่งชุดข้อมูล

#### การทำความสะอาดข้อมูล

ใช้ฟังก์ชัน `clean_thai_text` เพื่อทำความสะอาดข้อความภาษาไทย โดยการลบอักขระพิเศษและการทำให้ข้อความเป็นมาตรฐาน

```python
from utils.thai_nlp_utils import clean_thai_text

text = "ข้อความภาษาไทยที่ต้องการประมวลผล"
cleaned_text = clean_thai_text(text)
print(cleaned_text)
```

#### การตัดคำ

ใช้ฟังก์ชัน `tokenize_thai_text` เพื่อทำการตัดคำในข้อความภาษาไทย

```python
from utils.thai_nlp_utils import tokenize_thai_text

tokens = tokenize_thai_text(cleaned_text)
print(tokens)
```

#### การแบ่งชุดข้อมูล

ใช้ฟังก์ชัน `prepare_dataset_splits` เพื่อแบ่งชุดข้อมูลเป็นชุดฝึกสอน ชุดตรวจสอบ และชุดทดสอบ

```python
from utils.thai_nlp_utils import prepare_dataset_splits

# สมมติว่ามี DataFrame ที่มีคอลัมน์ 'text' และ 'label'
df = pd.DataFrame({
    'text': ["ข้อความที่ 1", "ข้อความที่ 2", "ข้อความที่ 3"],
    'label': [0, 1, 0]
})

train_df, val_df, test_df = prepare_dataset_splits(df, text_col='text', label_col='label')
print(f"Train set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")
print(f"Test set: {test_df.shape}")
```

### 2. การเตรียมโมเดล

เลือกโมเดลที่เหมาะสมสำหรับงานที่ต้องการ เช่น BERT, RoBERTa, หรือ GPT และโหลดโทเคไนเซอร์และโมเดล

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
NUM_LABELS = 3  # จำนวนคลาสที่ต้องการจำแนก

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
```

### 3. การสร้าง Dataset

สร้างคลาส Dataset สำหรับการฝึกสอนโมเดล

```python
from torch.utils.data import Dataset

class ThaiTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# สร้าง Dataset สำหรับการฝึกสอน การตรวจสอบ และการทดสอบ
train_dataset = ThaiTextClassificationDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_length=128)
val_dataset = ThaiTextClassificationDataset(val_df['text'].values, val_df['label'].values, tokenizer, max_length=128)
test_dataset = ThaiTextClassificationDataset(test_df['text'].values, test_df['label'].values, tokenizer, max_length=128)
```

### 4. การฝึกสอนโมเดล

กำหนดพารามิเตอร์การฝึกสอนและเริ่มการฝึกสอนโมเดล

```python
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

train_results = trainer.train()
print(f"Training metrics: {train_results}")
```

### 5. การประเมินผลโมเดล

ประเมินผลโมเดลบนชุดทดสอบและสร้างรายงานการจำแนก

```python
test_results = trainer.evaluate(test_dataset)
print(f"Test metrics: {test_results}")

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(test_df['label'], preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(test_df['label'], preds))
```

### 6. การบันทึกโมเดลไปยัง Hugging Face

บันทึกโมเดลและโทเคไนเซอร์ไปยัง Hugging Face Hub

```python
from utils.thai_nlp_utils import save_model_to_hf
from huggingface_hub import notebook_login

HF_TOKEN = "your_huggingface_token"  # รับโทเค็นจาก https://huggingface.co/settings/tokens
MODEL_REPO_NAME = "your-username/thai-text-classification"

notebook_login()

save_model_to_hf(model, tokenizer, MODEL_REPO_NAME, HF_TOKEN)
```

### 7. การทดสอบโมเดลที่บันทึก

ทดสอบโมเดลที่บันทึกด้วยข้อความตัวอย่าง

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification", 
    model=MODEL_REPO_NAME,
    tokenizer=MODEL_REPO_NAME
)

sample_texts = [
    "หนังเรื่องนี้สนุกมากๆ",
    "อาหารที่ร้านนี้รสชาติดีมาก"
]

for text in sample_texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Prediction: {result}\n")
```

## การฝึกสอนบน Google Colab Pro

โมเดลทั้งหมดสามารถฝึกสอนได้อย่างมีประสิทธิภาพบน Google Colab Pro โดยใช้ GPU หรือ TPU โดยทั่วไปแนะนำการตั้งค่าดังนี้:

1. **เลือก Runtime ที่เหมาะสม**:
   - สำหรับโมเดลขนาดใหญ่: ควรเลือก GPU T4 หรือ P100
   - สำหรับการฝึกสอนที่ใช้เวลานาน: อัพเกรดเป็น Colab Pro+ เพื่อเวลาและทรัพยากรที่มากขึ้น

2. **การจัดการหน่วยความจำ**:
   - ใช้ Gradient Accumulation สำหรับโมเดลขนาดใหญ่
   - เพิ่มประสิทธิภาพด้วย Mixed Precision Training (fp16)

3. **การบันทึกผลลัพธ์**:
   - บันทึกโมเดลไปยัง Google Drive หรือ Hugging Face Hub
   - ใช้ callbacks เพื่อบันทึกจุดตรวจสอบ (checkpoints) ระหว่างการฝึกสอน

ดูตัวอย่างการใช้งานเพิ่มเติมได้ใน notebooks ที่อยู่ในโฟลเดอร์ `models/`

## อ้างอิงและแหล่งข้อมูล

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) - ไลบรารีประมวลผลภาษาไทย
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - ไลบรารีสำหรับโมเดล Transformer
- [WangchanBERTa](https://github.com/vistec-AI/thai2transformers) - โมเดลภาษาไทยโดย VISTEC-AI
- [Thai National Language Processing Consortium](https://www.nectec.or.th/corpus/) - แหล่งข้อมูลคลังข้อความภาษาไทย

## การมีส่วนร่วมและการสนับสนุน

เรายินดีรับการมีส่วนร่วมจากชุมชน! หากคุณมีข้อเสนอแนะ พบบัก หรือต้องการเพิ่มฟีเจอร์ใหม่ กรุณาสร้าง Issue หรือ Pull Request ใน GitHub repository

## ลิขสิทธิ์

โปรเจกต์นี้เผยแพร่ภายใต้ลิขสิทธิ์ MIT License - ดูรายละเอียดเพิ่มเติมในไฟล์ LICENSE
