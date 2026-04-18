mkdir -p data/babyslm/{lexical,syntactic}

# Download dev
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/dev.zip -P data/babyslm/lexical
wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/dev.zip -P data/babyslm/syntactic
unzip data/babyslm/lexical/dev.zip -d data/babyslm/lexical
unzip data/babyslm/syntactic/dev.zip -d data/babyslm/syntactic
# remove zip
rm data/babyslm/lexical/dev.zip
rm data/babyslm/syntactic/dev.zip
# fix name
mv data/babyslm/syntactic/dev_16 data/babyslm/syntactic/dev