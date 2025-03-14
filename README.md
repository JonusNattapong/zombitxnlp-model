# ZombitX NLP - โมเดลประมวลผลภาษาไทย

ระบบนี้คือกรอบการพัฒนาที่ครอบคลุมสำหรับการฝึกสอนและพัฒนาโมเดลภาษาไทยขั้นสูงสำหรับงานประมวลผลภาษาธรรมชาติหลากหลายรูปแบบ

## งานประมวลผลภาษาที่รองรับ

### 1. การจำแนกข้อความ (Text Classification)
ZombitX NLP รองรับการสร้างโมเดลจำแนกประเภทข้อความภาษาไทยได้หลากหลายรูปแบบ ได้แก่:
- **การวิเคราะห์ความรู้สึก (Sentiment Analysis)**: จำแนกอารมณ์และความรู้สึกในข้อความ เช่น บวก กลาง ลบ
- **การจำแนกหัวข้อ (Topic Classification)**: แยกประเภทเนื้อหาตามหัวข้อหรือหมวดหมู่
- **การจับเจตนา (Intent Detection)**: ระบุเจตนาหรือจุดประสงค์ของผู้เขียน
- **การแบ่งหมวดหมู่เนื้อหา (Content Categorization)**: แยกประเภทเนื้อหาตามลักษณะหรือวัตถุประสงค์

โมเดลจำแนกข้อความมีขั้นตอนการพัฒนาที่สำคัญ ได้แก่:
1. การเตรียมข้อมูล: ทำความสะอาดข้อความ แบ่งชุดข้อมูล
2. การเลือกโครงสร้างโมเดล: เช่น BERT หรือ RoBERTa ที่ปรับแต่งสำหรับภาษาไทย
3. การฝึกสอนและปรับแต่ง: Fine-tune บนข้อมูลเฉพาะทาง
4. การประเมินผล: ใช้เมตริกที่เหมาะสมเช่น Accuracy, F1-score

### 2. การจำแนกโทเค็น (Token Classification)
งานจำแนกข้อความระดับคำหรือโทเค็น รวมถึง:
- **การรู้จำชื่อเฉพาะ (Named Entity Recognition)**: ระบุชื่อบุคคล สถานที่ องค์กร วันเวลา
- **การระบุชนิดคำ (Part-of-Speech Tagging)**: จำแนกชนิดคำ เช่น คำนาม คำกริยา คำคุณศัพท์
- **การแบ่งกลุ่มคำ (Chunking)**: จัดกลุ่มคำที่มีความสัมพันธ์กัน
- **การสกัดคำสำคัญ (Keyphrase Extraction)**: ดึงคำหรือวลีสำคัญจากข้อความ

การพัฒนาโมเดลจำแนกโทเค็นมีความท้าทายเฉพาะสำหรับภาษาไทยเนื่องจากไม่มีการแบ่งคำที่ชัดเจน จึงต้องมีการจัดการเรื่องการตัดคำและการแมปป้ายกำกับอย่างถูกต้อง

### 3. การตอบคำถาม (Question Answering)
ระบบสามารถพัฒนาโมเดลตอบคำถามภาษาไทยได้หลายรูปแบบ:
- **การตอบคำถามแบบสกัดคำตอบ (Extractive QA)**: ดึงคำตอบจากบริบทที่กำหนด
- **การตอบคำถามจากตาราง (Table QA)**: ค้นหาคำตอบจากข้อมูลในรูปแบบตาราง
- **การตอบคำถามจากบริบท (Context-based QA)**: ให้คำตอบโดยอ้างอิงจากบริบทที่กำหนด

โมเดลตอบคำถามภาษาไทยใช้สถาปัตยกรรม Encoder-Decoder พร้อมกลไก Attention ที่ช่วยให้โมเดลสามารถเข้าใจความสัมพันธ์ระหว่างคำถามและเนื้อหาที่เกี่ยวข้อง

### 4. การแปลภาษา (Translation)
ZombitX NLP รองรับการพัฒนาโมเดลแปลภาษาคุณภาพสูง:
- **การแปลไทย-อังกฤษ**: แปลจากภาษาไทยเป็นภาษาอังกฤษ
- **การแปลอังกฤษ-ไทย**: แปลจากภาษาอังกฤษเป็นภาษาไทย
- **รองรับหลายภาษา**: สามารถขยายไปยังการแปลภาษาอื่นๆ ได้

โมเดลแปลภาษาใช้สถาปัตยกรรม Transformer ที่ได้รับการปรับแต่งเฉพาะสำหรับภาษาไทย พร้อมเทคนิค Beam Search เพื่อปรับปรุงคุณภาพการแปล

### 5. การสรุปความ (Summarization)
ความสามารถในการสรุปเนื้อหาภาษาไทย:
- **การสรุปแบบสกัด (Extractive)**: เลือกประโยคสำคัญจากต้นฉบับ
- **การสรุปแบบสร้างใหม่ (Abstractive)**: สร้างข้อความสรุปใหม่ที่กระชับและเข้าใจง่าย
- **การสร้างพาดหัว (Headline Generation)**: สร้างพาดหัวข่าวหรือหัวข้อที่น่าสนใจ

โมเดลสรุปความใช้เทคนิค Copy Mechanism และ Coverage Mechanism เพื่อสร้างบทสรุปที่สมบูรณ์และไม่ซ้ำซ้อน

### 6. การสกัดคุณลักษณะ (Feature Extraction)
เครื่องมือสำหรับการสกัดคุณลักษณะจากข้อความภาษาไทย:
- **Word Embeddings**: การแปลงคำเป็นเวกเตอร์
- **Contextual Embeddings**: การแทนคำโดยคำนึงถึงบริบท
- **Semantic Similarity**: การวัดความคล้ายคลึงเชิงความหมาย
- **Text Representations**: การแทนข้อความในรูปแบบเวกเตอร์

การสกัดคุณลักษณะช่วยในการเตรียมข้อมูลสำหรับงาน Downstream Tasks ต่างๆ และใช้เทคนิค Self-supervised Learning

### 7. การสร้างข้อความ (Text Generation)
ความสามารถในการสร้างข้อความภาษาไทย:
- **การสร้างเนื้อหาสร้างสรรค์**: สร้างข้อความที่มีความสร้างสรรค์และหลากหลาย
- **การเพิ่มปริมาณข้อมูล**: สร้างข้อมูลเพิ่มเติมสำหรับการฝึกสอนโมเดล
- **การเติมเต็มเนื้อหา**: สร้างหรือเติมเต็มเนื้อหาที่ไม่สมบูรณ์
- **การสร้างบทสนทนา**: สร้างบทสนทนาที่เป็นธรรมชาติ

โมเดลสร้างข้อความใช้สถาปัตยกรรมแบบ GPT และสามารถควบคุมรูปแบบ (Style) และหัวข้อ (Topic) ของข้อความที่สร้างได้

## สถาปัตยกรรมโมเดล

ZombitX NLP รองรับสถาปัตยกรรมโมเดลหลากหลายรูปแบบ:

- **โมเดลฐาน BERT/RoBERTa**: โมเดลพื้นฐานที่มีประสิทธิภาพสูง เหมาะกับงานจำแนกข้อความและโทเค็น
- **โมเดล Transformer แบบ Encoder-Decoder**: เหมาะสำหรับงานแปลภาษาและสรุปความ
- **โมเดลแบบ GPT**: สำหรับการสร้างข้อความที่มีคุณภาพสูง
- **โมเดล BiLSTM-CRF**: เหมาะกับการจำแนกลำดับ เช่น NER และ POS Tagging
- **โมเดลเฉพาะทางสำหรับภาษาไทย**: โมเดลที่ได้รับการปรับแต่งเป็นพิเศษสำหรับภาษาไทย

## เครื่องมือและยูทิลิตี้

กรอบการทำงานนี้มาพร้อมกับเครื่องมือที่ครบครัน:

- **การเตรียมข้อมูลและทำความสะอาดข้อความภาษาไทย**: ฟังก์ชัน clean_thai_text และเครื่องมืออื่นๆ
- **การตัดคำภาษาไทยขั้นสูง**: ฟังก์ชัน tokenize_thai_text ที่รองรับหลายเอนจิ้น
- **การเตรียมชุดข้อมูลสำหรับงาน NLP ต่างๆ**: เช่น prepare_dataset_splits, prepare_qa_dataset
- **ไปป์ไลน์สำหรับการฝึกสอนและประเมินผล**: ฟังก์ชันสำหรับการฝึกสอนและประเมินผลโมเดลแต่ละประเภท
- **การส่งออกโมเดลสำหรับใช้งานจริง**: ฟังก์ชัน export_model_for_production
- **การเพิ่มปริมาณข้อมูลเฉพาะสำหรับภาษาไทย**: เช่น advanced_thai_augmentation

## เริ่มต้นใช้งาน

### ข้อกำหนดเบื้องต้น

```bash
pip install transformers datasets evaluate pythainlp sentencepiece torch pandas sklearn rouge nltk
```

### การติดตั้งและการตั้งค่า

1. โคลนโปรเจกต์:
```bash
git clone https://github.com/zombitx64/zombitxnlp-model.git
cd zombitxnlp-model
```

2. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

### การใช้งานพื้นฐาน

```python
from utils.thai_nlp_utils import clean_thai_text, tokenize_thai_text, create_model_for_task

# ทำความสะอาดและเตรียมข้อความภาษาไทย
text = "ข้อความภาษาไทยที่ต้องการประมวลผล"
cleaned_text = clean_thai_text(text)
tokens = tokenize_thai_text(cleaned_text)

# สร้างโมเดลสำหรับงานเฉพาะทาง
model = create_model_for_task(
    task_type="classification",
    pretrained_model_name="airesearch/wangchanberta-base-att-spm-uncased",
    num_labels=3
)
```

## ตัวอย่างการใช้งาน

### 1. การจำแนกข้อความ (Text Classification)

```python
# นำเข้าไลบรารีที่จำเป็น
from utils.thai_nlp_utils import create_model_for_task, get_training_args_for_task
from transformers import AutoTokenizer, Trainer

# โหลดโทเคไนเซอร์และสร้างโมเดล
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
model = create_model_for_task(
    task_type="classification",
    pretrained_model_name="airesearch/wangchanberta-base-att-spm-uncased",
    num_labels=3  # จำนวนคลาสที่ต้องการจำแนก
)

# กำหนดการฝึกสอน
training_args = get_training_args_for_task(
    task_type="classification", 
    output_dir="./results",
    batch_size=16,
    num_epochs=5
)

# เริ่มการฝึกสอน
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

### 2. การรู้จำชื่อเฉพาะ (Named Entity Recognition)

```python
# นำเข้าไลบรารีที่จำเป็น
from utils.thai_nlp_utils import prepare_token_classification_data, create_model_for_task

# เตรียมข้อมูล
texts = ["สมชายไปกรุงเทพเมื่อวานนี้", "วันนี้อากาศที่เชียงใหม่ดีมาก"]
token_labels = [[0, 0, 1, 1, 2, 2], [2, 2, 0, 0, 1, 1, 0]]  # 0=O, 1=LOC, 2=TIME

# เตรียมข้อมูลสำหรับการฝึกสอน
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
dataset = prepare_token_classification_data(texts, token_labels, tokenizer)

# สร้างโมเดล
model = create_model_for_task(
    task_type="token_classification",
    pretrained_model_name="airesearch/wangchanberta-base-att-spm-uncased",
    num_labels=3
)
```

### 3. การตอบคำถาม (Question Answering)

```python
# นำเข้าไลบรารีที่จำเป็น
from utils.thai_nlp_utils import prepare_qa_dataset, create_model_for_task

# กำหนดข้อมูล
questions = ["กรุงเทพมหานครเป็นเมืองหลวงของประเทศอะไร?"]
contexts = ["กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย"]
answers = [{'text': ['ประเทศไทย'], 'answer_start': [66]}]

# เตรียมข้อมูล
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
dataset = prepare_qa_dataset(questions, contexts, answers, tokenizer)

# สร้างโมเดล
model = create_model_for_task(
    task_type="qa",
    pretrained_model_name="airesearch/wangchanberta-base-att-spm-uncased"
)
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
