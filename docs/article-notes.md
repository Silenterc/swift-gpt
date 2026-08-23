# Chapter 1 - Understanding LLMs
- Compared to traditional ML models, they are initially not trained on labeled data = self-supervised learning (it generates its own "labels")
- Pretraining -> Foundation Model -> Finetuning -> Capable model (classification, translation...)
    - The main ones are Instruction x Classification finetuning
- Idea: In the article sketch the GPT architecture and highlight what will I be implementing from scratch
- BERT uses Encoder and is useful for masked word prediction -> text classification, sentiment analysis
- GPT uses Decoder and generates text (next word prediction) - it does not use Encoders at all, even modern ones

# Chapter 2 - Working with text data
- No re.split alternative in Swift
- It is not scalable to load the whole training dataset to do text -> tokenizer -> ids -> embedding...
    - Probably a good idea to tokenize -> write to files -> load files with tokens
- Simple tokenId embedding isnt enough because every tokenId from different parts of a text would get mapped to the same place
    - The model therefore wouldnt know the order of words
    - Can fix this by relative x absolute positional embeddings
- So the data flow will be: .txt training data -> LocalFileReader -> strings -> Tokenizer -> tokenIds -> TokenFileWriter -> .bin token files -> TokenDataset -> DataLoader + sliding-window sampling -> (inputs, targets) -> Embedding
- Storing and loading individual .txt files isnt as scalable as I thought, modern datasets use Parquet files or json
- I will use ~ 18 000 books from Project Gutenberg when I will be training the model - mostly novels, then short stories, science fiction, philosophy, history, science, essays
- I want around 2B tokens so I can aim for 100-150M parameters





















# Chapter 3 - Coding attention mechanisms

# Chapter 4 - Implementing a GPT model from scratch to generate text

# Chapter 5 - Pretraining on unlabeled data

# Chapter 6 - Fine-tuning for classification

# Chapter 7 - Fine-tuning to follow instructions


