import sys


map_file = sys.argv[1]
raw_test_file = sys.argv[2]
output_file = sys.argv[3]

date_set = ('year_0_number', 'year_1_number', 'year_2_number', 'year_3_number', 'month_0_number', 'month_0_name', 'month_1_name', 'day_0_number', 'day_1_number')


def replace_date(tok):
    if tok == 'year_0_number':
        tok = 'year_0'
    elif tok == 'year_1_number':
        tok = 'year_1'
    elif tok == 'year_2_number':
        tok = 'year_2'
    elif tok == 'year_3_number':
        tok = 'year_3'
    elif tok == 'month_0_number':
        tok = 'month_0'
    elif tok == 'month_0_name':
        tok = 'month_0'
    elif tok == 'month_1_name':
        tok = 'month_1'
    elif tok == 'day_0_number':
        tok = 'day_0'
    elif tok == 'day_1_number':
        tok = 'day_1'

    return tok


mapping_list = list()
with open(map_file) as f:
    map_list = f.readlines()
    print(len(map_list))
    for line in map_list:
        line = line.strip()
        if line != '{}':
            line = line[1:-1]
            entity_dict = dict()
            if ',' in line:
                entity_list = line.split('",')
                for entity in entity_list:
                    entity = entity.split(':')
                    anon = entity[0].strip()[1:-1]
                    if entity[1].strip()[-1] == '"':
                        deanon = entity[1].strip()[1:-1].lower()
                    else:
                        deanon = entity[1].strip()[1:].lower()
                    entity_dict[anon] = deanon
            else:
                entity = line.split(':')
                anon = entity[0].strip()[1:-1]
                deanon = entity[1].strip()[1:-1].lower()
                entity_dict[anon] = deanon
            # print(entity_dict)
            mapping_list.append(entity_dict)
        else:
            mapping_list.append([])

print(len(mapping_list))

with open(raw_test_file) as f:
    output_list = f.readlines()
    all_sent_list = list()
    for index, line in enumerate(output_list):
        entities = mapping_list[index]
        if not len(entities):
            all_sent_list.append(line)
            continue
        sent_list = line.strip().split(' ')
        # print(entities)
        new_sent = ''
        for tok in sent_list:
            if tok in date_set:
                tok = replace_date(tok)
                print(tok)
            if tok in entities.keys():
                deanon = entities[tok]
                new_sent += deanon + ' '
            else:
                new_sent += tok + ' '
        new_sent += '\n'
        # print(new_sent)
        all_sent_list.append(new_sent)

with open(output_file, 'w') as out:
    for sent in all_sent_list:
        out.write(sent)

# print(all_sent_list)
