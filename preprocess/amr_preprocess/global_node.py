import argparse

def add_global_node(dirt, name):

	f1 = open(dirt + name + ".amr", "r")
	f2 = open(dirt + name + ".grh", "r")
	h1 = open(dirt + name + ".amr_g", "w")
	h2 = open(dirt + name + ".grh_g", "w")
	index_list = []
	node = "gnode"
	for line in f1:
		toks = line.strip().split()
		for t in toks:
			h1.write(str(t) + ' ')
		h1.write(str(node) + '\n')
		index_list.append(len(toks))
	i = 0
	for line in f2:
		deps = line.strip().split()
		for d in deps:
			h2.write(str(d)+ ' ')
		index = index_list[i]
		for j in range(index):
			d1 = '(' + str(index) + ',' + str(j) + ',g)'
			h2.write(d1+' ')
		d3 = '(' + str(index) + ',' + str(index) + ',s)'
		h2.write(d3+'\n')
		i += 1


def gen_amrgrh(dirt, name):

	file_4 = dirt + name + ".amr_g"
	file_5 = dirt + name + ".grh_g"
	file_6 = dirt + name + ".amrgrh_g"
	h4 = open(file_4, "r")
	h5 = open(file_5, "r")
	h6 = open(file_6, "w")
	toks = []
	deps = []
	for line in h4:
		seg = line.strip().split()
		toks.append(seg)
	for line in h5:
		seg = line.strip().split()
		deps.append(seg)
	L = len(toks)
	if len(toks) != len(deps):
		print("error3")
	for i in range(L):
		tok = toks[i]
		dep = deps[i]
		for j in range(len(tok)):
			h6.write(str(tok[j]) + ' ')
		h6.write("\t")
		for j in range(len(dep)):
			h6.write(str(dep[j]) + ' ')
		h6.write("\n")


if __name__ == '__main__':
		# Parse input
	parser = argparse.ArgumentParser(description="process AMR with the global node")
	parser.add_argument('--input_dir', type=str, help='input dir')
	# name_list = ["train", "test", "val"]
	# dirt = "en2cs/"
	args = parser.parse_args()
	name_list = ["train", "test", "dev"]
	for name in name_list:
		add_global_node(args.input_dir, name)
		gen_amrgrh(args.input_dir, name)















