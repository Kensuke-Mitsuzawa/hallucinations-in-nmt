import os
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel

# Paths to your model and data
model_dir = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/models/wmt18_de-en"  # Replace with the path to your wmt18_de-en directory
model_file = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/models/checkpoint_best.pt"  # Replace with the path to your checkpoint file
sentencepiece_model = "/media/ariel/mitsuzaw/code/hallucinations-in-nmt/sentencepiece_models/sentencepiece.joint.bpe.model"  # Replace with your SentencePiece model

# Load SentencePiece model for tokenization
sp = spm.SentencePieceProcessor()
sp.load(sentencepiece_model)

# Load the pre-trained Fairseq model
translator = TransformerModel.from_pretrained(
    model_dir,
    checkpoint_file=model_file,
    data_name_or_path=model_dir,
    bpe="sentencepiece",
    sentencepiece_model=sentencepiece_model,
)

# Sample German input text
german_text = "Das ist ein Beispielsatz, den wir übersetzen möchten."

# Tokenize German text using SentencePiece
tokenized_input = sp.encode(german_text, out_type=str)
print(f"Tokenized input: {tokenized_input}")

# Translate tokenized input
translated_tokens = translator.translate(" ".join(tokenized_input))

# Detokenize the output
print(translated_tokens.split())
translated_text = sp.decode(translated_tokens.split())
print(f"Translated text: {translated_text}")
