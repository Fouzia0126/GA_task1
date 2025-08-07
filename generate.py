from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_dir="./gpt2-finetuned", max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    generated = generate_text(prompt)
    print("\nGenerated Text:\n", generated)
