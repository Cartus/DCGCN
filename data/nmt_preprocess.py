def add_global_node(dirt, name):

	f1 = open(dirt + name + ".en.tok", "r")
	f2 = open(dirt + name + ".en.deps", "r")
	f3 = open(dirt + name + ".en.tokdeps", "r")
	h1 = open(dirt + name + ".en.tok_gd", "w")
	h2 = open(dirt + name + ".en.deps_gd", "w")
	h3 = open(dirt + name + ".en.tokdeps_gd", "w")
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
	for line in f3:
		seg = line.strip().split('\t')
		toks = seg[0].split()
		deps = seg[1].split()
		for t in toks:
			h3.write(str(t) + ' ')
		h3.write(str(node) + '\t')
		for d in deps:
			h3.write(str(d) + ' ')
		index = len(toks)
		for j in range(index):
			d1 = '(' + str(index) + ',' + str(j) + ',g)'
			h3.write(d1+' ')
		d3 = '(' + str(index) + ',' + str(index) + ',s)'
		h3.write(d3+'\n')


def dep_to_levi(dirt, name, Sequential=False):

	if name == "train":
		file_1 = "en2cs/news-commentary-v11.cs-en.clean.tok.en"
		file_2 = "en2cs/news-commentary-v11.cs-en.clean.deprels.en"
		file_3 = "en2cs/news-commentary-v11.cs-en.clean.heads.en"
	elif name == "test":
		file_1 = "test/newstest2016-encs-src.tok.en"
		file_2 = "test/newstest2016-encs-src.deprels.en"
		file_3 = "test/newstest2016-encs-src.heads.en"
	elif name == "val":
		file_1 = "dev/newstest2015-encs-src.tok.en"
		file_2 = "dev/newstest2015-encs-src.deprels.en"
		file_3 = "dev/newstest2015-encs-src.heads.en"

	file_4 = dirt + name + ".en.tok"
	file_5 = dirt + name + ".en.deps"
	f1 = open(file_1, "r")
	f2 = open(file_2, "r")
	f3 = open(file_3, "r")
	h1 = open(file_4, "w")
	h2 = open(file_5, "w")

	tok_list = []
	dep_list = []
	head_list = []

	for line in f1:
		seg = line.rstrip().split()
		tok_list.append(seg)

	for line in f2:
		seg = line.rstrip().split()
		dep_list.append(seg)

	for line in f3:
		seg = line.rstrip().split()
		head_list.append(seg)

	N = len(tok_list)
	if len(tok_list) != len(dep_list):
		print("error1")
	for i in range(N):
		tok = tok_list[i]
		dep = dep_list[i]
		head = head_list[i]
		M = len(tok)
		if M == 0:
			print(i)
		if len(tok) != len(dep):
			print("error2")
		for j in range(M):
			h1.write(str(tok[j]) + ' ')
			if Sequential:
				if j+1 < M:
					d1 = '(' + str(j) + ',' + str(j+1) + ',f)'
					h2.write(d1 + ' ')
					d2 = '(' + str(j+1) + ',' + str(j) + ',b)'
					h2.write(d2 + ' ')
		for k in range(M):
			h1.write(str(dep[k]) + ' ')
			g1 = '(' + str(k) + ',' + str(k) + ',s)'
			h2.write(g1 + ' ')
			g2 = '(' + str(k+M) + ',' + str(k+M) + ',s)'
			h2.write(g2 + ' ')
			index = int(head[k])
			if index != 0:
				g3 = '(' + str(index-1) + ',' + str(k+M) + ',d)'
				h2.write(g3 + ' ')
				g4 = '(' + str(k+M) + ',' + str(index-1) + ',r)'
				h2.write(g4 + ' ')

			g5 = '(' + str(k+M) + ',' + str(k) + ',d)'
			h2.write(g5 + ' ')
			g6 = '(' + str(k) + ',' + str(k+M) + ',r)'
			h2.write(g6 + ' ')
		h1.write('\n')
		h2.write('\n')

	print(N)

def gen_tokdeps(dirt, name):

	file_4 = dirt + name + ".en.tok"
	file_5 = dirt + name + ".en.deps"
	file_6 = dirt + name + ".en.tokdeps"
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
	add_global = True
	Sequential = True
	name_list = ["train", "test", "val"]
	dirt = "en2cs/"
	for name in name_list:
		dep_to_levi(dirt, name, Sequential)
		if name != "train":
			gen_tokdeps(dirt, name)
		if add_global:
			add_global_node(dirt, name)















