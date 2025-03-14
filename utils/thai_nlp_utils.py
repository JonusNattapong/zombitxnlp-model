import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pythainlp import word_tokenize
from pythainlp.util import normalize
from transformers import AutoTokenizer

def clean_thai_text(text):
    """Clean Thai text by removing special characters and normalizing"""
    if not isinstance(text, str):
        return ""
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_thai_text(text, engine="newmm"):
    """Tokenize Thai text using PyThaiNLP"""
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text, engine=engine)
    return tokens

def prepare_dataset_splits(df, text_col, label_col=None, test_size=0.1, val_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets"""
    # First split: train + validation vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_col] if label_col else None
    )
    
    # Second split: train vs validation
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state, 
            stratify=train_val[label_col] if label_col else None
        )
        return train, val, test
    return train_val, None, test

def save_model_to_hf(model, tokenizer, model_name, auth_token):
    """Save model and tokenizer to Hugging Face Hub"""
    model.push_to_hub(model_name, use_auth_token=auth_token)
    tokenizer.push_to_hub(model_name, use_auth_token=auth_token)
    print(f"Model and tokenizer saved to Hugging Face Hub: {model_name}")

def load_pretrained_thai_tokenizer(model_name="airesearch/wangchanberta-base-att-spm-uncased"):
    """Load a pretrained Thai tokenizer"""
    return AutoTokenizer.from_pretrained(model_name)

def evaluate_model_predictions(y_true, y_pred, task_type="classification"):
    """Evaluate model predictions based on task type"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    if task_type == "classification":
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted')
        }
    # Add more evaluation types as needed
    
    return results

def augment_thai_text(text, augmentation_type="synonym", probability=0.2):
    """Augment Thai text using different strategies
    
    Args:
        text (str): Original Thai text
        augmentation_type (str): Type of augmentation - "synonym", "random_swap", "random_delete"
        probability (float): Probability of applying augmentation to each token
    
    Returns:
        str: Augmented text
    """
    if not isinstance(text, str) or not text.strip():
        return text
        
    tokens = tokenize_thai_text(text)
    
    if augmentation_type == "random_swap" and len(tokens) > 1:
        # Randomly swap adjacent tokens
        for i in range(len(tokens) - 1):
            if np.random.random() < probability:
                tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    
    elif augmentation_type == "random_delete":
        # Randomly delete tokens
        tokens = [t for t in tokens if np.random.random() > probability]
    
    # For synonym replacement, we would need a Thai synonym dictionary
    # This is a placeholder for that functionality
    
    return ''.join(tokens)

def prepare_multilabel_dataset(df, text_col, label_cols, test_size=0.1, val_size=0.1, random_state=42):
    """Prepare dataset for multi-label classification
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_col (str): Column name containing text
        label_cols (list): List of column names for each label
        test_size (float): Proportion of data to use for test set
        val_size (float): Proportion of data to use for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: train + validation vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs validation
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state
        )
        return train, val, test
    return train_val, None, test

def batch_predict(model, tokenizer, texts, device="cuda", batch_size=16, max_length=128):
    """Run batch prediction on a list of texts
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        texts (list): List of texts to predict
        device (str): Device to run inference on
        batch_size (int): Batch size for inference
        max_length (int): Maximum sequence length
    
    Returns:
        list: Predictions for each text
    """
    if not texts:
        return []
        
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
    
    return all_predictions

def export_model_for_production(model, tokenizer, output_dir):
    """Export model and tokenizer for production deployment
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        output_dir (str): Directory to save model and tokenizer
        
    Returns:
        str: Path to saved model
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save a sample script for inference
    with open(os.path.join(output_dir, "predict.py"), "w") as f:
        f.write("""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(text, model, tokenizer, device="cpu"):
    model.to(device)
    model.eval()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()
    
    return {
        "class": pred_class,
        "probability": probs[0][pred_class].item()
    }

if __name__ == "__main__":
    model_path = "."  # Current directory 
    model, tokenizer = load_model(model_path)
    
    # Example text
    text = "ทดสอบระบบภาษาไทย"
    result = predict(text, model, tokenizer)
    print(f"Prediction: {result}")
""")
    
    return output_dir

def create_balanced_sampler(dataset, label_col):
    """Create a sampler for imbalanced datasets
    
    Args:
        dataset: Dataset with labels
        label_col (str): Name of column containing labels
        
    Returns:
        torch.utils.data.WeightedRandomSampler: Balanced sampler
    """
    from torch.utils.data import WeightedRandomSampler
    
    labels = dataset[label_col].values
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

# Enhanced text augmentation with Thai-specific techniques
def advanced_thai_augmentation(text, technique="synonym", strength=0.2, max_length=None):
    """Advanced data augmentation for Thai text
    
    Args:
        text (str): Original Thai text
        technique (str): Augmentation technique - "synonym", "swap", "delete", "insert", "typo", "tts", "backtranslation"
        strength (float): Amount of augmentation to apply (0.0-1.0)
        max_length (int): Maximum sequence length to return
        
    Returns:
        str: Augmented Thai text
    """
    if not isinstance(text, str) or not text.strip():
        return text
        
    tokens = tokenize_thai_text(text)
    result = tokens.copy()
    
    # Select tokens to modify based on strength
    num_to_change = max(1, int(len(tokens) * strength))
    indices = np.random.choice(range(len(tokens)), num_to_change, replace=False)
    
    if technique == "swap":
        # Advanced swap (not just adjacent tokens)
        for i in indices:
            if i < len(tokens) - 1:
                swap_idx = np.random.randint(0, len(tokens))
                result[i], result[swap_idx] = result[swap_idx], result[i]
                
    elif technique == "delete":
        # Delete random tokens
        result = [t for i, t in enumerate(tokens) if i not in indices]
        
    elif technique == "insert":
        # Insert common Thai words or characters
        common_thai_words = ["ที่", "และ", "ของ", "ใน", "การ", "มี", "ไม่", "เป็น", "จะ"]
        for i in sorted(indices, reverse=True):
            insert_word = np.random.choice(common_thai_words)
            if i < len(result):
                result.insert(i, insert_word)
    
    elif technique == "typo":
        # Simulate Thai keyboard typos
        char_maps = {
            'ก': ['ข', 'ถ'],
            'า': ['ส', 'ว'],
            'ม': ['บ', 'น'],
            # Add more mappings for Thai keyboard layout
        }
        
        for i in indices:
            if i < len(result) and len(result[i]) > 0:
                char_idx = np.random.randint(0, len(result[i]))
                char = result[i][char_idx]
                if char in char_maps:
                    replacement = np.random.choice(char_maps[char])
                    result[i] = result[i][:char_idx] + replacement + result[i][char_idx+1:]
    
    # Join tokens back into text
    augmented_text = ''.join(result)
    
    # Trim to max_length if specified
    if max_length and len(augmented_text) > max_length:
        augmented_text = augmented_text[:max_length]
        
    return augmented_text

# Prepare data for token classification tasks (NER, POS tagging)
def prepare_token_classification_data(texts, token_labels, tokenizer, max_length=128):
    """Prepare dataset for token classification tasks like NER or POS tagging
    
    Args:
        texts (list): List of input texts
        token_labels (list): List of token labels (list of lists)
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Dictionary with input_ids, attention_mask, labels
    """
    input_ids = []
    attention_masks = []
    labels_list = []
    
    for text, doc_labels in zip(texts, token_labels):
        # Tokenize text
        tokens = tokenize_thai_text(text)
        
        # Check if labels match tokens
        if len(tokens) != len(doc_labels):
            raise ValueError(f"Mismatch between tokens ({len(tokens)}) and labels ({len(doc_labels)})")
        
        # Map to subwords with BERT/RoBERTa tokenizers
        bert_tokens = []
        bert_labels = []
        
        for token, label in zip(tokens, doc_labels):
            # Tokenize the word and count # of subwords
            subwords = tokenizer.tokenize(token)
            
            # Add the tokenized word to the final tokenized word list
            bert_tokens.extend(subwords)
            
            # Add the same label to each subword of the token
            bert_labels.extend([label] + [-100] * (len(subwords) - 1))
        
        # Account for [CLS] and [SEP] tokens
        bert_tokens = [tokenizer.cls_token] + bert_tokens + [tokenizer.sep_token]
        bert_labels = [-100] + bert_labels + [-100]
        
        # Convert tokens to IDs
        encoding = tokenizer.convert_tokens_to_ids(bert_tokens)
        
        # Pad or truncate sequences
        if len(encoding) < max_length:
            encoding = encoding + [tokenizer.pad_token_id] * (max_length - len(encoding))
            bert_labels = bert_labels + [-100] * (max_length - len(bert_labels))
            attention_mask = [1] * len(bert_labels) + [0] * (max_length - len(bert_labels))
        else:
            encoding = encoding[:max_length]
            bert_labels = bert_labels[:max_length]
            attention_mask = [1] * max_length
        
        input_ids.append(encoding)
        attention_masks.append(attention_mask)
        labels_list.append(bert_labels)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels_list)
    }

# Prepare data for question answering tasks
def prepare_qa_dataset(questions, contexts, answers=None, tokenizer=None, max_length=384, doc_stride=128):
    """Prepare dataset for question answering tasks
    
    Args:
        questions (list): List of questions
        contexts (list): List of contexts/passages
        answers (list, optional): List of answer dictionaries with 'text' and 'answer_start'
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        doc_stride (int): Stride size for splitting long contexts
        
    Returns:
        dict: Dataset formatted for QA task
    """
    from transformers import squad_convert_examples_to_features
    
    # Create examples in the SQuAD format
    examples = []
    
    for i, (question, context) in enumerate(zip(questions, contexts)):
        example = {
            'id': str(i),
            'question': question,
            'context': context,
            'answers': answers[i] if answers else {'text': [], 'answer_start': []}
        }
        examples.append(example)
    
    # Convert to features
    features = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_length,
        doc_stride=doc_stride,
        max_query_length=64,
        is_training=answers is not None
    )
    
    # Convert to PyTorch tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    
    if answers:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "start_positions": all_start_positions,
            "end_positions": all_end_positions
        }
    else:
        all_example_ids = [f.example_index for f in features]
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "example_ids": all_example_ids
        }

# Prepare data for translation tasks
def prepare_translation_dataset(source_texts, target_texts, src_tokenizer, tgt_tokenizer, 
                               src_max_length=128, tgt_max_length=128):
    """Prepare dataset for translation tasks
    
    Args:
        source_texts (list): List of source language texts
        target_texts (list): List of target language texts
        src_tokenizer: Tokenizer for source language
        tgt_tokenizer: Tokenizer for target language
        src_max_length (int): Maximum source sequence length
        tgt_max_length (int): Maximum target sequence length
        
    Returns:
        dict: Dataset formatted for translation
    """
    # Tokenize source texts
    source_encodings = src_tokenizer(
        source_texts,
        max_length=src_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize target texts
    with src_tokenizer.as_target_tokenizer():
        target_encodings = tgt_tokenizer(
            target_texts,
            max_length=tgt_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    # Create labels (shifting for next-token prediction)
    labels = target_encodings.input_ids.clone()
    # Set pad tokens to -100 to ignore in loss calculation
    labels[labels == tgt_tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": source_encodings.input_ids,
        "attention_mask": source_encodings.attention_mask,
        "labels": labels
    }

# Prepare data for summarization tasks
def prepare_summarization_dataset(texts, summaries, tokenizer, text_max_length=512, summary_max_length=128):
    """Prepare dataset for text summarization tasks
    
    Args:
        texts (list): List of input texts to summarize
        summaries (list): List of reference summaries
        tokenizer: HuggingFace tokenizer
        text_max_length (int): Maximum input text length
        summary_max_length (int): Maximum summary length
        
    Returns:
        dict: Dataset formatted for summarization
    """
    # Tokenize input texts
    input_encodings = tokenizer(
        texts,
        max_length=text_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize summaries
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            summaries,
            max_length=summary_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    # Create labels (shifting for next-token prediction)
    labels = target_encodings.input_ids.clone()
    # Set pad tokens to -100 to ignore in loss calculation
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "decoder_input_ids": target_encodings.input_ids,
        "labels": labels
    }

# Evaluate token classification models (NER, POS tagging)
def evaluate_token_classification(true_labels, pred_labels, label_list):
    """Evaluate token classification models (NER, POS tagging)
    
    Args:
        true_labels (list): List of true labels (list of lists)
        pred_labels (list): List of predicted labels (list of lists)
        label_list (list): List of label names
        
    Returns:
        dict: Evaluation metrics
    """
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    
    # Convert numeric labels to string labels
    true_labels_str = [[label_list[l] if l != -100 else 'O' for l in doc] for doc in true_labels]
    pred_labels_str = [[label_list[l] if l != -100 else 'O' for l in doc] for doc in pred_labels]
    
    # Calculate metrics
    results = {
        "precision": precision_score(true_labels_str, pred_labels_str),
        "recall": recall_score(true_labels_str, pred_labels_str),
        "f1": f1_score(true_labels_str, pred_labels_str),
        "report": classification_report(true_labels_str, pred_labels_str, output_dict=True)
    }
    
    return results

# Evaluate translation and summarization models
def evaluate_generation_metrics(references, hypotheses, task="translation"):
    """Evaluate generation metrics for translation or summarization
    
    Args:
        references (list): List of reference texts
        hypotheses (list): List of generated texts
        task (str): Task type ("translation" or "summarization")
        
    Returns:
        dict: Evaluation metrics
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge import Rouge
    
    results = {}
    
    # For both translation and summarization
    # Prepare data for BLEU calculation
    tokenized_refs = [[tokenize_thai_text(ref)] for ref in references]
    tokenized_hyps = [tokenize_thai_text(hyp) for hyp in hypotheses]
    
    # Calculate BLEU score
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smooth)
    results["bleu"] = bleu_score
    
    # For summarization, add ROUGE scores
    if task == "summarization":
        rouge = Rouge()
        rouge_scores = rouge.get_scores([' '.join(h) for h in tokenized_hyps], 
                                       [' '.join(r[0]) for r in tokenized_refs], 
                                       avg=True)
        results["rouge"] = rouge_scores
    
    return results

# Create model architecture for different tasks
def create_model_for_task(task_type, pretrained_model_name, num_labels=None, config_overrides=None):
    """Create model architecture for specific NLP task
    
    Args:
        task_type (str): Type of task - "classification", "token_classification", "qa", 
                         "translation", "summarization", "generation"
        pretrained_model_name (str): Base pretrained model name
        num_labels (int, optional): Number of labels for classification tasks
        config_overrides (dict, optional): Additional configuration overrides
        
    Returns:
        model: Appropriate model for the task
    """
    from transformers import (
        AutoConfig, 
        AutoModelForSequenceClassification, 
        AutoModelForTokenClassification, 
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM
    )
    
    config = AutoConfig.from_pretrained(pretrained_model_name)
    
    # Apply any config overrides
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)
    
    # Set number of labels if applicable
    if num_labels is not None:
        config.num_labels = num_labels
    
    # Create appropriate model based on task
    if task_type == "classification":
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
    
    elif task_type == "token_classification":
        return AutoModelForTokenClassification.from_pretrained(pretrained_model_name, config=config)
    
    elif task_type == "qa":
        return AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name, config=config)
    
    elif task_type in ["translation", "summarization"]:
        return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name, config=config)
    
    elif task_type == "generation":
        return AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")

# Create training arguments for different NLP tasks
def get_training_args_for_task(task_type, output_dir, batch_size=16, num_epochs=3, learning_rate=2e-5):
    """Get training arguments optimized for specific NLP tasks
    
    Args:
        task_type (str): Type of task - "classification", "token_classification", "qa", 
                         "translation", "summarization", "generation"
        output_dir (str): Directory to save model
        batch_size (int): Training batch size
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        
    Returns:
        TrainingArguments: Configured training arguments
    """
    from transformers import TrainingArguments
    
    # Common arguments for all tasks
    common_args = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 50,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "learning_rate": learning_rate,
        "fp16": True
    }
    
    # Task-specific arguments
    task_specific_args = {}
    
    if task_type == "classification":
        task_specific_args = {
            "metric_for_best_model": "f1"
        }
    
    elif task_type == "token_classification":
        task_specific_args = {
            "metric_for_best_model": "f1"
        }
    
    elif task_type == "qa":
        task_specific_args = {
            "metric_for_best_model": "f1"
        }
    
    elif task_type in ["translation", "summarization"]:
        task_specific_args = {
            "predict_with_generate": True,
            "generation_max_length": 128,
            "generation_num_beams": 4,
            "metric_for_best_model": "bleu" if task_type == "translation" else "rouge"
        }
    
    elif task_type == "generation":
        task_specific_args = {
            "metric_for_best_model": "perplexity"
        }
    
    # Combine common and task-specific arguments
    args = {**common_args, **task_specific_args}
    
    return TrainingArguments(**args)
