SUBJECTS=("f")
#SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
PRIOR="phpbb"
TV_TYPE="samsung"

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json /local/smart-tv-user-study/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json /local/dictionaries/passwords/${PRIOR}.db /local/dictionaries/english/wikipedia.db
done

PRIOR="rockyou-5gram"
for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner /local/smart-tv-user-study/subject-${s}/${TV_TYPE}_passwords.json /local/smart-tv-user-study/subject-${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json /local/dictionaries/passwords/${PRIOR}.db /local/dictionaries/english/wikipedia.db
done

