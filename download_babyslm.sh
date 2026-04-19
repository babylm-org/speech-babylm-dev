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

# download test (might be long)
# wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/lexical/test.zip -P <DATA_LOCATION>/babyslm/lexical
# wget https://cognitive-ml.fr/downloads/baby-slm/evaluation_sets/syntactic/test.zip -P <DATA_LOCATION>/babyslm/syntactic

# unzip <DATA_LOCATION>/babyslm/lexical/test.zip -d <DATA_LOCATION>/babyslm/lexical
# unzip <DATA_LOCATION>/babyslm/syntactic/test.zip -d <DATA_LOCATION>/babyslm/syntactic