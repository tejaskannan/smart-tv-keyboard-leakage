SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
TV_TYPE="samsung"
USER_BASE="<FILL-IN>"
DICTIONARY_BASE="<FILL-IN>"

PRIOR="phpbb"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/${TV_TYPE}_passwords.json --output-file ${USER_BASE}/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt
done

PRIOR="rockyou-5gram"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/${TV_TYPE}_passwords.json --output-file ${USER_BASE}/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt
done

PRIOR="phpbb"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/${TV_TYPE}_passwords.json --output-file ${USER_BASE}/subject-${s}/no_directions_recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt --ignore-directions
done

PRIOR="rockyou-5gram"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file ${USER_BASE}/subject-${s}/${TV_TYPE}_passwords.json --output-file ${USER_BASE}/subject-${s}/no_directions_recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior ${DICTIONARY_BASE}/${PRIOR}.db --english-prior ${DICTIONARY_BASE}/wikipedia.db --zip-prior ${DICTIONARY_BASE}/zip_codes.txt --ignore-directions
done

