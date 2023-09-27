PARTS=( $(seq 0 11) )
PRIOR="phpbb"
TV_TYPE="samsung"
KEYBOARD_TYPE="samsung"
INPUT_BASE="<FILL-IN>"
DICTIONARY_BASE="<FILL-IN>"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${INPUT_BASE}/${KEYBOARD_TYPE}-passwords/part_${s}/${TV_TYPE}_passwords.json --output-file ${INPUT_BASE}/${KEYBOARD_TYPE}-passwords/part_${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt --ignore-suboptimal
done
