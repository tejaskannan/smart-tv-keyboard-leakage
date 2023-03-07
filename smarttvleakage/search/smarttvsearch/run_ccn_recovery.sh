SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner /local/smart-tv-user-study/subject-${s}/credit_card_details.json /local/dictionaries/credit_cards/zip_codes.txt /local/smart-tv-user-study/subject-${s}/recovered_credit_card_details.json
done
