PARTS=( $(seq 0 19) )
PRIOR="rockyou-5gram"
TV_TYPE="appletv"
BASE="/local/smart-tv-benchmarks"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner ${BASE}/${TV_TYPE}-passwords/part_${s}/${TV_TYPE}_passwords.json ${BASE}/${TV_TYPE}-passwords/part_${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json /local/dictionaries/passwords/${PRIOR}.db /local/dictionaries/english/wikipedia.db
done
