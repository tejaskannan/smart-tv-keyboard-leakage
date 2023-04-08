SUBJECTS=("e")
OUTPUT_FOLDER="/local/samsung/ccn_search_comparison"
OUTPUT_FILE_NAME="exhaustive.json"

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/credit_card_details.json --output-file ${OUTPUT_FOLDER}/subject-${s}/${OUTPUT_FILE_NAME} --password-prior /local/dictionaries/passwords/phpbb.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt
done
