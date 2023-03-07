PARTS=( $(seq 0 19) )
PRIOR="rockyou-5gram"
TV_TYPE="samsung"
BASE="/local/smart-tv-benchmarks"

for s in "${PARTS[@]}"
do
    java smarttvsearch.SearchRunner ${BASE}/${TV_TYPE}-passwords/part_${s}/${TV_TYPE}_passwords.json /local/dictionaries/passwords/${PRIOR}.db ${BASE}/${TV_TYPE}-passwords/part_${s}/recovered_${TV_TYPE}_passwords_${PRIOR}.json
done
