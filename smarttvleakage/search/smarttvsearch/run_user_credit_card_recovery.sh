SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
USER_BASE="<FILL-IN>"
DICTIONARY_BASE="<FILL-IN>"

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/credit_card_details.json --output-file ${USER_BASE}/subject-${s}/recovered_credit_card_details.json --password-prior ${DICTIONARY_BASE}/phpbb.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt
done

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/credit_card_details.json --output-file ${USER_BASE}/subject-${s}/exhaustive.json --password-prior ${DICTIONARY_BASE}/phpbb.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt --use-exhaustive
done
