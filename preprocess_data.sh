cat model_bpe/raw_data/train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q > model_bpe/preprocessed_data/train.de.p

cat model_bpe/raw_data/train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q > model_bpe/preprocessed_data/train.en.p

perl moses_scripts/train-truecaser.perl --model model_bpe/preprocessed_data/tm.de --corpus model_bpe/preprocessed_data/train.de.p

perl moses_scripts/train-truecaser.perl --model model_bpe/preprocessed_data/tm.en --corpus model_bpe/preprocessed_data/train.en.p

cat model_bpe/preprocessed_data/train.de.p | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.de > model_bpe/preprocessed_data/train.de 

cat model_bpe/preprocessed_data/train.en.p | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.en > model_bpe/preprocessed_data/train.en

cat model_bpe/raw_data/valid.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.de > model_bpe/preprocessed_data/valid.de

cat model_bpe/raw_data/valid.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.en > model_bpe/preprocessed_data/valid.en

cat model_bpe/raw_data/test.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.de > model_bpe/preprocessed_data/test.de

cat model_bpe/raw_data/test.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.en > model_bpe/preprocessed_data/test.en

cat model_bpe/raw_data/tiny_train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.de > model_bpe/preprocessed_data/tiny_train.de

cat model_bpe/raw_data/tiny_train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_bpe/preprocessed_data/tm.en > model_bpe/preprocessed_data/tiny_train.en


rm model_bpe/preprocessed_data/train.de.p
rm model_bpe/preprocessed_data/train.en.p

# BPE
BPE_CODES="model_bpe/preprocessed_data/bpe_codes"
# Number of operations
MERGE_OPS=16000

BPE_VOCAB="model_bpe/prepared_data/dict"
BPE_VOCAB_EN="$BPE_VOCAB.en"
BPE_VOCAB_DE="$BPE_VOCAB.de"

BPE_INPUT_TRAIN="model_bpe/preprocessed_data/train"
BPE_INPUT_TRAIN_EN="$BPE_INPUT_TRAIN.en"
BPE_INPUT_TRAIN_DE="$BPE_INPUT_TRAIN.de"

BPE_INPUT_TINY_TRAIN="model_bpe/preprocessed_data/tiny_train"
BPE_INPUT_TINY_TRAIN_EN="$BPE_INPUT_TINY_TRAIN.en"
BPE_INPUT_TINY_TRAIN_DE="$BPE_INPUT_TINY_TRAIN.de"

BPE_OUTPUT_TRAIN="model_bpe/preprocessed_data/train.bpe"
BPE_OUTPUT_TRAIN_EN="$BPE_OUTPUT_TRAIN.en"
BPE_OUTPUT_TRAIN_DE="$BPE_OUTPUT_TRAIN.de"

BPE_OUTPUT_TINY_TRAIN="model_bpe/preprocessed_data/tiny_train.bpe"
BPE_OUTPUT_TINY_TRAIN_EN="$BPE_OUTPUT_TINY_TRAIN.en"
BPE_OUTPUT_TINY_TRAIN_DE="$BPE_OUTPUT_TINY_TRAIN.de"

BPE_INPUT_VAL="model_bpe/preprocessed_data/valid"
BPE_INPUT_VAL_EN="$BPE_INPUT_VAL.en"
BPE_INPUT_VAL_DE="$BPE_INPUT_VAL.de"
BPE_OUTPUT_VAL="model_bpe/preprocessed_data/valid.bpe"
BPE_OUTPUT_VAL_EN="$BPE_OUTPUT_VAL.en"
BPE_OUTPUT_VAL_DE="$BPE_OUTPUT_VAL.de"

BPE_INPUT_TEST="model_bpe/preprocessed_data/test"
BPE_INPUT_TEST_EN="$BPE_INPUT_TEST.en"
BPE_INPUT_TEST_DE="$BPE_INPUT_TEST.de"

BPE_OUTPUT_TEST="model_bpe/preprocessed_data/test"
BPE_OUTPUT_TEST_EN="$BPE_OUTPUT_TEST.en"
BPE_OUTPUT_TEST_DE="$BPE_OUTPUT_TEST.de"

## Learn joint BPE and vocab
subword-nmt learn-joint-bpe-and-vocab --input $BPE_INPUT_TRAIN_DE $BPE_INPUT_TRAIN_EN -s $MERGE_OPS -o $BPE_CODES --write-vocabulary $BPE_VOCAB_DE $BPE_VOCAB_EN

## Apply
### Train
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_EN --vocabulary-threshold 1 < $BPE_INPUT_TRAIN_EN > $BPE_OUTPUT_TRAIN_EN
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_DE --vocabulary-threshold 1 < $BPE_INPUT_TRAIN_DE > $BPE_OUTPUT_TRAIN_DE

subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_EN --vocabulary-threshold 1 < $BPE_INPUT_TINY_TRAIN_EN > $BPE_OUTPUT_TINY_TRAIN_EN
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_DE --vocabulary-threshold 1 < $BPE_INPUT_TINY_TRAIN_DE > $BPE_OUTPUT_TINY_TRAIN_DE
## Validation
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_EN --vocabulary-threshold 1 < $BPE_INPUT_VAL_EN > $BPE_OUTPUT_VAL_EN
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_DE --vocabulary-threshold 1 < $BPE_INPUT_VAL_DE > $BPE_OUTPUT_VAL_DE
## Test
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_DE --vocabulary-threshold 1 < $BPE_INPUT_TEST_DE > $BPE_OUTPUT_TEST_DE
subword-nmt apply-bpe -c $BPE_CODES --vocabulary $BPE_VOCAB_EN --vocabulary-threshold 1 < $BPE_INPUT_TEST_EN > $BPE_OUTPUT_TEST_EN
# END BPE

python preprocess.py --target-lang en --source-lang de --dest-dir model_bpe/prepared_data/ --train-prefix $BPE_OUTPUT_TRAIN --valid-prefix $BPE_OUTPUT_VAL --test-prefix $BPE_OUTPUT_TEST --tiny-train-prefix $BPE_OUTPUT_TINY_TRAIN --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
