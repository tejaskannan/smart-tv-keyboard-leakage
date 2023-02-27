SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner /local/smart-tv-user-study/subject-${s}/samsung_passwords.json /local/dictionaries/passwords/ /local/smart-tv-user-study/subject-${s}/recovered_samsung_passwords_rockyou.json
done
