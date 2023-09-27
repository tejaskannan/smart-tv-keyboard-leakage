PARTS=( $(seq 0 11) )
INPUT_BASE="<FILL-IN>"
DICTIONARY_BASE="<FILL-IN>"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${BASE}/part_${s}/credit_card_details.json --output-file ${BASE}/part_${s}/recovered_credit_card_details.json --password-prior ${DICTIONARY_BASE}/phpbb.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt
done
