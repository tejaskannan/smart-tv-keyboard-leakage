PARTS=( $(seq 0 11) )
PRIOR="phpbb"
TV_TYPE="samsung"
BASE="/local/smart-tv-benchmarks"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${BASE}/${TV_TYPE}-passwords/part_${s}/${TV_TYPE}_passwords.json --output-file ${BASE}/${TV_TYPE}-passwords/part_${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt --ignore-suboptimal
done

echo "APPLETV"
TV_TYPE="appletv"
for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${BASE}/${TV_TYPE}-passwords/part_${s}/${TV_TYPE}_passwords.json --output-file ${BASE}/${TV_TYPE}-passwords/part_${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt --ignore-suboptimal
done

