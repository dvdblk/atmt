cat model_v1/raw_data/train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q > model_v1/preprocessed_data/train.de.p

cat model_v1/raw_data/train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q > model_v1/preprocessed_data/train.en.p

perl moses_scripts/train-truecaser.perl --model model_v1/preprocessed_data/tm.de --corpus model_v1/preprocessed_data/train.de.p

perl moses_scripts/train-truecaser.perl --model model_v1/preprocessed_data/tm.en --corpus model_v1/preprocessed_data/train.en.p

cat model_v1/preprocessed_data/train.de.p | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.de > model_v1/preprocessed_data/train.de 

cat model_v1/preprocessed_data/train.en.p | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.en > model_v1/preprocessed_data/train.en

cat model_v1/raw_data/valid.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.de > model_v1/preprocessed_data/valid.de

cat model_v1/raw_data/valid.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.en > model_v1/preprocessed_data/valid.en

cat model_v1/raw_data/test.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.de > model_v1/preprocessed_data/test.de

cat model_v1/raw_data/test.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.en > model_v1/preprocessed_data/test.en

cat model_v1/raw_data/tiny_train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.de > model_v1/preprocessed_data/tiny_train.de

cat model_v1/raw_data/tiny_train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model model_v1/preprocessed_data/tm.en > model_v1/preprocessed_data/tiny_train.en


rm model_v1/preprocessed_data/train.de.p
rm model_v1/preprocessed_data/train.en.p

# BPE
BPE_INPUT_EN="model_v1/preprocessed_data/train.en"
BPE_INPUT_DE="model_v1/preprocessed_data/train.de"
CODES_EN="model_v1/prepared_data/codes.en"
CODES_DE="model_v1/prepared_data/codes.de"
MERGE_OPS=16000
subword-nmt learn-bpe -s $MERGE_OPS < $BPE_INPUT_EN > $CODES_EN
subword-nmt learn-bpe -s $MERGE_OPS < $BPE_INPUT_DE > $CODES_DE

subword-nmt apply-bpe -c $CODES_EN < $BPE_INPUT_EN > model_v1/prepared_data/train.en
subword-nmt apply-bpe -c $CODES_DE < $BPE_INPUT_DE > model_v1/prepared_data/train.de
# END BPE

python preprocess.py --target-lang en --source-lang de --dest-dir model_v1/prepared_data/ --train-prefix model_v1/preprocessed_data/train --valid-prefix model_v1/preprocessed_data/valid --test-prefix model_v1/preprocessed_data/test --tiny-train-prefix model_v1/preprocessed_data/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
