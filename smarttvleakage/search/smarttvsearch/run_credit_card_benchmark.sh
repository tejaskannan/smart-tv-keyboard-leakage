PARTS=( $(seq 0 11) )
BASE="/local/smart-tv-benchmarks/credit_cards"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${BASE}/part_${s}/credit_card_details.json --output-file ${BASE}/part_${s}/recovered_credit_card_details.json --password-prior /local/dictionaries/password/phpbb.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt --ignore-suboptimal
done
