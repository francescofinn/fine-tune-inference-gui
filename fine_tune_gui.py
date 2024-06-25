import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

class FineTuneApp:
    def __init__(self, root):
        # Initialise labels, buttons & text inputs
        self.root = root
        self.root.title("Fine-Tune LLM")

        self.label = tk.Label(root, text="Select Data File:")
        self.label.pack(pady=10)

        self.file_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.file_button.pack(pady=5)

        self.train_button = tk.Button(root, text="Start Fine-Tuning", command=self.fine_tune)
        self.train_button.pack(pady=20)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=10)

        self.input_label = tk.Label(root, text="Enter Prompt for Inference:")
        self.input_label.pack(pady=10)

        self.input_bar = tk.Entry(root, width=50)
        self.input_bar.pack(pady=30)

        self.infer_button = tk.Button(root, text="Generate Text", command=self.generate_text)
        self.infer_button.pack(pady=20)

        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.pack(pady=10)

        self.data_path = ""
        self.model_path = './fine_tuned_model' # Once model is fine-tuned, it's saved to working directory

        self.tokenizer = None
        self.model = None

    def browse_file(self):
        self.data_path = filedialog.askopenfilename()
        self.status_label.config(text=f"Selected File: {self.data_path}")

    def fine_tune(self):
        if not self.data_path:
            messagebox.showerror("Error", "Please select a data file.")
            return
        
        self.status_label.config(text="Fine-tuning in progress...")

        try:
            # Load dataset from text file
            with open(self.data_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                dataset = Dataset.from_dict({'text': lines}) # Initialise Dataset object

            # Verify number of dataset samples
            if len(dataset) == 0:
                raise ValueError("The dataset is empty.")
            
            # Split into train and test sets
            dataset = dataset.train_test_split(test_size=0.1)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']

            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
            tokenizer.pad_token = tokenizer.eos_token # Set pad_token to eos_token (end of sequence)
            
            def tokenize_function(examples):
                outputs = tokenizer(examples['text'], padding="max_length", truncation=True)
                outputs['labels'] = outputs['input_ids'].copy() # Use input_ids as labels
                return outputs
            
            tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
            tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
            
            training_args = TrainingArguments(
                output_dir='./results',
                eval_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets,
                eval_dataset=tokenized_eval_dataset
            )

            trainer.train()
            trainer.save_model(self.model_path) # Save fine-tuned model
            tokenizer.save_pretrained(self.model_path) # Save tokenizer
            self.status_label.config(text="Model fine-tuning complete.")
        
        except Exception as e:
            self.status_label.config(text="Error during fine-tuning.")
            messagebox.showerror("Error", str(e))

    def generate_text(self):
        if not self.tokenizer or not self.model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            except Exception as e:
                self.status_label.config(text="Error loading model.")
                messagebox.showerror("Error", str(e))
                return        
        
        prompt = self.input_bar.get()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt for inference.")
            return
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=50, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, generated_text)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x600")
    app = FineTuneApp(root)
    root.mainloop()
