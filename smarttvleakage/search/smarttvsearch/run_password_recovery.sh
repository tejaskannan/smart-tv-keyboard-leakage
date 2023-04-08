SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
TV_TYPE="samsung"

PRIOR="phpbb"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json --output-file /local/smart-tv-user-study/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt
done

PRIOR="rockyou-5gram"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json --output-file /local/smart-tv-user-study/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt
done

PRIOR="phpbb"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json --output-file /local/smart-tv-user-study/subject-${s}/no_directions_recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt --ignore-directions
done

PRIOR="rockyou-5gram"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json --output-file /local/smart-tv-user-study/subject-${s}/no_directions_recovered_${TV_TYPE}_passwords_${PRIOR}.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_cards/zip_codes.txt --ignore-directions
done

