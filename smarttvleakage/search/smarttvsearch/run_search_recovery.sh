#SUBJECTS=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j")
SUBJECTS=("f" "i")
PRIOR="phpbb"

for s in "${SUBJECTS[@]}"
do
    java smarttvsearch.SearchRunner /local/smart-tv-user-study/subject-${s}/web_searches.json /local/smart-tv-user-study/subject-${s}/recovered_web_searches.json /local/dictionaries/passwords/${PRIOR}.db /local/dictionaries/english/wikipedia.db
done
