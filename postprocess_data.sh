infile=$1
outfile=$2
lang=$3

cat $infile | perl moses_scripts/detruecase.perl | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile

