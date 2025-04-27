from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.train(
    "--input=../datasets/corpus.txt\
    model_prefix=petition_bpe\
    --vocab_size=8000
    model_type="bpe"
)