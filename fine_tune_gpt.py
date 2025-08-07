import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    # Paths
    train_file = "data/train.txt"
    output_dir = "./gpt2-finetuned"

    # Load tokenizer and model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load dataset
    train_dataset = load_dataset(train_file, tokenizer)

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved at {output_dir}")

if __name__ == "__main__":
    main()
