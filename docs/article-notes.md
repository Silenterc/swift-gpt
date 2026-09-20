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
- The heart of the algo
- Softmax is used for normalization
- Each embedding gets transformed into Query, Key and Value vectors by a linear transformation
- Query vector means what I am looking for, key means what they can offer me, and if query and key are similar (their dot product is big), they attend
- These similarity scores get transformed into attention weights via softmax normalization (and beforehand scaled by / srt (dim))
- Then we add the values weighted by those weights to get the context vector, which then gets used to update the embedding
$$Z = A \times V = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$
- In order for the LLM to not "cheat" and see all (the following) tokens ahead of time, they get -inf scored - called masking/causal attention
- K x Q can get very large for large context window sizes
- We can use dropout to avoid overfitting - randomly drop data during training, in our case attention weights from the attention w matrix
- For the article probably go with general attention info -> Q,K,V + softmax explanation -> Causal Attention -> Multi Head

# Chapter 4 - Implementing a GPT model from scratch to generate text
- GPT model = Input -> Embedding -> Transfrmer blocks -> Output layers -> Output
- GPT 2 config of 124 params has: 50 257 vocab size, 1024 context length, 768 emb dim, 12 att heads, 12 layers, 0.1 dropout, false qkv bias
- Transformer block = LayerNorm1 -> Attention -> Dropout -> shortcut -> LayerNorm2 -> Feed forward -> Dropout -> shortcut
- LayerNorm normalizes the outputs of a NN to have a mean of 0 and variance 1, for better training (handle gradients better)
- Feed forward = Linear(emb_dim, 4*emb_dim) -> GELU (act. function) -> Linear(4*emb_dim, emb_dim)
- GELU/Activation functions introduce non-linearity, otherwise it would all just be one big linear projection, and the model wouldnt learn much nuance
- Shortcut means adding the input of some layer/network to the output of it
    - Helps with the vanishing gradient problem (earlier layers have smaller gradients)
- 

# Chapter 5 - Pretraining on unlabeled data
- For my data, I will need to split the dataset into train/ and val/, with a split like 99/1%
    - Then tokenize it and load it to/from there
- 




# Chapter 6 - Fine-tuning for classification

# Chapter 7 - Fine-tuning to follow instructions


