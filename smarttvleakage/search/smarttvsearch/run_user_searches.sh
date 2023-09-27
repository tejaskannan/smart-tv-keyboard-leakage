SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
PRIOR="phpbb"
USER_BASE="<FILL-IN>"
DICTIONARY_BASE="<FILL-IN>"


for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/web_searches.json --output-file ${USER_BASE}/subject-${s}/recovered_web_searches.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt
done


for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/web_searches.json --output-file ${USER_BASE}/subject-${s}/forced_recovered_web_searches.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt --force-suggestions
done
