SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
PRIOR="phpbb"

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner --input-file /local/smart-tv-user-study/subject-${s}/web_searches.json --output-file /local/smart-tv-user-study/subject-${s}/forced_recovered_web_searches.json --password-prior /local/dictionaries/passwords/${PRIOR}.db --english-prior /local/dictionaries/english/wikipedia.db --zip-prior /local/dictionaries/credit_card/zip_codes.txt
done
