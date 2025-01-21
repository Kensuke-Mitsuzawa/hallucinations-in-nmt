import torch
from fairseq.models.transformer import TransformerModel

# Paths to your model and data
model_dir = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/models/wmt18_de-en"  # Replace with your directory
model_file = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/models/checkpoint_best.pt"  # Replace with your model file
model_sentencepiece = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/sentencepiece_models/sentencepiece.joint.bpe.model"

# Load the pre-trained Fairseq model
model = TransformerModel.from_pretrained(
    model_dir,
    checkpoint_file=model_file,
    data_name_or_path=model_dir,
    bpe="sentencepiece",
    sentencepiece_model=model_sentencepiece
)

# Sentence to get embeddings for
english_sentence = "This is an example sentence."

# Tokenize and convert to indices
tokenized = model.bpe.encode(english_sentence).split()
token_indices = [model.task.source_dictionary.index(token) for token in tokenized]

# Convert token indices to embeddings
with torch.no_grad():
    token_tensor = torch.tensor(token_indices).unsqueeze(0)  # Batch size of 1
    embeddings = model.models[0].decoder.embed_tokens(token_tensor)

print(f"Tokenized Sentence: {tokenized}")
print(f"Embeddings Shape: {embeddings.shape}")  # Shape: (1, sequence_length, embedding_dim)
