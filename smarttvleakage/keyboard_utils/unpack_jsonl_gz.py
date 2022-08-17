from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.audio import Move

def read_moves(file_path: str):
	output=[]
	for i in read_jsonl_gz(file_path):
	    temp = []
	    temp.append(i["word"])
	    for j in i["move_seq"]:
	        temp.append(Move(num_moves=int(j["num_moves"]),end_sound=j["sound"]))
	    output.append(temp)
	return output