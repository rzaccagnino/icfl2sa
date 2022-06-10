import time
import argparse

from factorizations import ICFL_recursive

from microdict import mdict

# Given a word 'w' return the suffix array (of w) 'SA_w'
def sorting_suffixes_via_icfl(w):

    start_time = time.time()

    # Step 1: initialize hash
    suffix_dict, lcp_sorted_suffixes, mask_index, bound = step_1(w)

    # Step 2: fill hash
    SA_w = step_2(suffix_dict, lcp_sorted_suffixes, mask_index,w)

    print("--- %s time seconds ---" % (time.time() - start_time))

    return SA_w


# STEP 1: initialize hash
def step_1(w):

    SA_w = []

    # compute icfl
    facts = ICFL_recursive(w)
    bound = compute_bound(facts)
    mask_index = compute_mask_index(facts)

    # build distinct local suffixes
    #distinct_local_suffixes,suffix_dict = build_distinct_local_suffixes(facts)
    suffix_dict = build_distinct_local_suffixes(facts)

    # sorting distinct local suffixes
    #sorted_distinct_local_suffixes, lcp_sorted_suffixes, sorted_suffixes_dict = sort_suffixes(distinct_local_suffixes)
    lcp_sorted_suffixes = sort_suffixes(suffix_dict)

    # build hash_table (sorted array + dictionary)

    suffix_dict = initialize_hash(suffix_dict, w, mask_index)

    return suffix_dict, lcp_sorted_suffixes, mask_index, bound

def compute_bound(facts):

    bound = 0
    for i in range(1, len(facts)):
        s = facts[i-1]+facts[i]
        l = len(s)
        if l > bound:
            bound = l

    return l

# Build distinct local suffixes
def build_distinct_local_suffixes(facts):

    suffix_dict = dict()
    #distinct_local_suffixes = []

    for fact in facts:
        fact_suffix_dict = {}
        #suffixes = []

        for j in range(0, len(fact)):
            #suffixes.append(fact[j:])
            fact_suffix_dict[fact[j:]] = []

        suffix_dict.update(fact_suffix_dict)
        #distinct_local_suffixes = distinct_local_suffixes + suffixes

    #return list(set(distinct_local_suffixes)),suffix_dict
    return suffix_dict


# Sort distinct local suffixes
def sort_suffixes(suffix_dict):

    lcp_sorted_suffixes = [0]

    sorted_keys = sorted(suffix_dict)

    for j in range (1, len(sorted_keys)):
        lcp_sorted_suffixes.append(LCP_length(sorted_keys[j-1],sorted_keys[j]))

    return lcp_sorted_suffixes


def LCP_length(X, Y):
    i = j = 0
    while i < len(X) and j < len(Y):
        if X[i] != Y[j]:
            break
        i = i + 1
        j = j + 1

    return len(X[:i])


def compute_mask_index(factors):

    mask_index = [0]
    for j in range(1, len(factors)):
        mask_index.append(mask_index[j-1] + len(factors[j-1]))

    return mask_index



def initialize_hash(suffix_dict, w, mask_index):


    if len(mask_index) == 1:

        for j in range(len(w)-1, -1,-1):
            suffix_dict[w[j:len(w)]].insert(0, j)

    else:

        # step 1
        for i in range(len(mask_index)-1,0,-1):

            for j in range(mask_index[i]-1,mask_index[i-1]-1,-1):

                suffix_dict[w[j:mask_index[i]]].insert(0,j)

        # step 2
        for j in range(len(w)-1, mask_index[-1]-1,-1):

            suffix_dict[w[j:len(w)]].insert(0, j)


    return suffix_dict


# STEP 2: Fill hash
def step_2(suffix_dict, lcp_sorted_suffixes, mask_index, w):

    sorted_keys = sorted(suffix_dict)
    speed_keys_array = speed_structure(sorted_keys, lcp_sorted_suffixes)

    len_speed = len(speed_keys_array)
    r_len_speed = range(len_speed)
    SA = []
    SA_group = []

    group = []
    for i in r_len_speed:

        current_speed = speed_keys_array[i]

        if len(current_speed) == 0 or lcp_sorted_suffixes[i] == 0:

            if len(group) > 1:
                #l = intelligent_merge(group)
                l = hope_merge(group)
                SA = SA + l
            else:
                SA = SA + SA_group

            group = []

            if i == len_speed - 1:
                current_key_list = sorted(current_speed)
                current_SA = suffix_dict[current_key_list[0]]

                SA = SA + current_SA
            continue

        current_key_list = sorted(current_speed)
        current_SA = suffix_dict[current_key_list[0]]

        len_dictionary = len(current_key_list)
        if len_dictionary == 1:
            continue
        else:
            r = range(1,len_dictionary)
            for index in r:
                SA_group, SA_hash = merge_hash_lists_prefix(current_SA, suffix_dict[current_key_list[index]], mask_index, w)
                current_SA = SA_group

            group.append(SA_group)

    return SA

def intelligent_merge(group):

    SA = []
    # Il gruppo ha sicuramente più di un elemento
    #group_dict = mdict.create("i32:i32")
    group_dict = dict()
    group_dict_key = dict()

    group_dict_is_taken = dict()

    main_lista = []
    keys_lista = []

    for i in range(len(group)):
        main_lista = main_lista + group[i]

    for j in range(len(main_lista)):

        #lista = group_dict[l[j]]
        lista = group_dict.get(main_lista[j])
        if lista == None:
            group_dict[main_lista[j]] = [j]
            keys_lista.append(main_lista[j])


        else:
            lista.append(j)
            group_dict[main_lista[j]] = lista

        group_dict_key[j] = main_lista[j]

    SA.append(keys_lista[0])
    for i in range(1,len(keys_lista)):
        key = keys_lista[i]

        lista = group_dict.get(key)

        if len(lista) > 1:

            for j in range(len(lista)):
                item = group_dict_key.get(lista[j]-1)

                taken = group_dict_is_taken.get(item)
                if taken == None:
                    SA.append(item)
                    group_dict_is_taken[item] = 1

            taken = group_dict_is_taken.get(key)
            if taken == None:
                SA.append(key)
                group_dict_is_taken[key] = 1

    return  SA

def hope_merge(group):

    SA = []

    # Il gruppo ha sicuramente più di un elemento
    #group_dict = mdict.create("i32:i32")
    group_dict = dict()
    group_dict_key = dict()

    group_dict_is_taken = dict()

    #main_lista = [16, 0, 1, 2, 9, 16, 12, 1, 5, 2, 9, 16, 13, 4, 2, 6, 9]

    main_lista = []
    keys_lista = []

    # Merge delle liste
    for i in range(len(group)):
        main_lista = main_lista + group[i]

    for j in range(len(main_lista)):

        #lista = group_dict[l[j]]
        lista = group_dict.get(main_lista[j])
        if lista == None:
            group_dict[main_lista[j]] = [j]
            keys_lista.append(main_lista[j])

        else:
            lista.append(j)
            group_dict[main_lista[j]] = lista

        group_dict_key[j] = main_lista[j]

    SA.append(keys_lista[0])
    group_dict_is_taken[keys_lista[0]] = 1

    for i in range(1,len(keys_lista)):
        key = keys_lista[i]

        lista = group_dict.get(key)

        if len(lista) > 1:

            for j in range(len(lista)):

                k = lista[j]-1
                not_taken_list = []
                while True:
                    item = group_dict_key.get(k)

                    taken = group_dict_is_taken.get(item)
                    if taken == None:
                        k = k-1
                        not_taken_list.insert(0,item)
                    else:
                        break

                for k in range(len(not_taken_list)):
                    SA.append(not_taken_list[k])
                    group_dict_is_taken[not_taken_list[k]] = 1

            taken = group_dict_is_taken.get(key)
            if taken == None:
                SA.append(key)
                group_dict_is_taken[key] = 1

    return  SA


def speed_structure(sorted_distinct_local_suffixes, lcp_sorted_suffixes):

    speed_keys_array = []

    r = range(0,len(sorted_distinct_local_suffixes))
    for i in r:
        speed_keys_array.append([sorted_distinct_local_suffixes[i]])

    r = range(len(sorted_distinct_local_suffixes)-1,-1,-1)
    for i in r:

        current_fact = sorted_distinct_local_suffixes[i]
        current_len = len(current_fact)

        j = i+1
        while j < len(lcp_sorted_suffixes):

            if lcp_sorted_suffixes[j] >= current_len:
                if len(speed_keys_array[j]) > 0:
                    speed_keys_array[j].append(sorted_distinct_local_suffixes[i])
                    speed_keys_array[i] = []
                j = j+1
            else:
                break
    '''
    i = 0
    while i < len(speed_keys_array):
        if len(speed_keys_array[i]) == 0:
            speed_keys_array.pop(i)
        else:
            i=i+1
    '''

    return speed_keys_array


def merge_hash_lists_prefix(SA_area, current_list, mask_index,w):

    SA = []
    SA_hash = mdict.create("i32:i32")
    #SA_hash = dict()

    start_index_to_compare = 0
    start_index_to_insert = 0

    len_SA = len(SA_area)
    len_current_list = len(current_list)
    while start_index_to_compare < len_SA and start_index_to_insert < len_current_list:

        index_to_insert = current_list[start_index_to_insert]
        index_to_compare = SA_area[start_index_to_compare]

        last_factor_index = len(mask_index) - 1
        fact_index_to_insert = fact_of(index_to_insert, mask_index)
        fact_index_to_compare = fact_of(index_to_compare, mask_index)


        if fact_index_to_insert == fact_index_to_compare:

            if fact_index_to_insert == last_factor_index:
                # il più piccolo viene dopo (nel SA)
                if index_to_insert < index_to_compare:
                    SA.append(index_to_compare)
                    SA_hash[index_to_compare] = len(SA) - 1
                    #SA.append(index_to_insert)
                    start_index_to_compare = start_index_to_compare + 1
                '''
                else:
                    SA.append(index_to_insert)
                    SA_hash[index_to_insert] = len(SA) - 1
                    #SA.append(index_to_compare)
                    start_index_to_insert = start_index_to_insert + 1
                '''
            else:

                # il più piccolo viene prima (nel SA)
                if index_to_insert < index_to_compare:
                    SA.append(index_to_insert)
                    SA_hash[index_to_insert] = len(SA) - 1
                    #SA.append(index_to_compare)
                    start_index_to_insert = start_index_to_insert + 1
                '''
                else:
                    SA.append(index_to_compare)
                    SA_hash[index_to_compare] = len(SA) - 1
                    #SA.append(index_to_insert)
                    start_index_to_compare = start_index_to_compare + 1
                '''
        elif fact_index_to_insert > fact_index_to_compare:

            if fact_index_to_insert != last_factor_index:
                # fact_index_to_compare viene prima di fact_index_to_insert
                suffix_to_insert = w[index_to_insert:mask_index[fact_index_to_insert+1]]
                suffix_to_compare = w[index_to_compare:mask_index[fact_index_to_compare+1]]

                if suffix_to_compare.startswith(suffix_to_insert):
                    SA.append(index_to_compare)
                    SA_hash[index_to_compare] = len(SA) - 1
                    #SA.append(index_to_insert)
                    start_index_to_compare = start_index_to_compare + 1
                else:

                    l = LCP_length(w[index_to_compare:],w[index_to_insert:])
                    #f1 = w[mask_index[fact_index_to_compare+1]:mask_index[fact_index_to_compare+2]]
                    #f2 = w[index_to_insert+len(suffix_to_compare):mask_index[fact_index_to_insert+1]]

                    i1 = index_to_compare + l
                    i2 = index_to_insert + l

                    if w[i1] < w[i2]:
                        SA.append(index_to_compare)
                        SA_hash[index_to_compare] = len(SA) - 1
                        start_index_to_compare = start_index_to_compare + 1
                    else:
                        SA.append(index_to_insert)
                        SA_hash[index_to_insert] = len(SA) - 1
                        start_index_to_insert = start_index_to_insert + 1

            else:
                # fact_index_to_insert viene prima di fact_index_to_compare
                SA.append(index_to_insert)
                SA_hash[index_to_insert] = len(SA) - 1
                #SA.append(index_to_compare)
                start_index_to_insert = start_index_to_insert + 1

        elif fact_index_to_compare > fact_index_to_insert:

            if fact_index_to_compare != last_factor_index:
                suffix_to_insert = w[index_to_insert:mask_index[fact_index_to_insert + 1]]
                suffix_to_compare = w[index_to_compare:mask_index[fact_index_to_compare + 1]]

                # fact_index_to_insert viene prima di fact_index_to_compare
                if suffix_to_insert.startswith(suffix_to_compare):
                    SA.append(index_to_insert)
                    SA_hash[index_to_insert] = len(SA) - 1
                    #SA.append(index_to_compare)
                    start_index_to_insert = start_index_to_insert + 1
                else:
                    l = LCP_length(w[index_to_insert:],w[index_to_compare:])
                    # f1 = w[mask_index[fact_index_to_compare+1]:mask_index[fact_index_to_compare+2]]
                    # f2 = w[index_to_insert+len(suffix_to_compare):mask_index[fact_index_to_insert+1]]

                    i1 = index_to_insert + l
                    i2 = index_to_compare + l

                    if w[i1] < w[i2]:
                        SA.append(index_to_insert)
                        SA_hash[index_to_insert] = len(SA) - 1
                        start_index_to_insert = start_index_to_insert + 1
                    else:

                        SA.append(index_to_compare)
                        SA_hash[index_to_compare] = len(SA) - 1
                        start_index_to_compare = start_index_to_compare + 1

            else:
                SA.append(index_to_compare)
                SA_hash[index_to_compare] = len(SA) - 1
                #SA.append(index_to_insert)
                start_index_to_compare = start_index_to_compare + 1

    # Ho completato l'inserimento di SA_area?
    #SA = SA + SA_area[start_index_to_compare:]
    for i in range(start_index_to_compare,len(SA_area)):
        SA.append(SA_area[i])
        SA_hash[SA_area[i]] = len(SA) - 1

    #SA = SA + current_list[start_index_to_insert:]
    for i in range(start_index_to_insert,len(current_list)):
        SA.append(current_list[i])
        SA_hash[current_list[i]] = len(SA) - 1
    return SA, SA_hash

def fact_of(index_to_insert,mask_index):
    fact_index = len(mask_index) - 1
    for i in range(1,len(mask_index)):
        if index_to_insert >= mask_index[i-1] and index_to_insert < mask_index[i]:
            return i-1

    return fact_index

def merge_hash_lists_external(SA_area, current_list, SA_hash):

    SA = []
    #SA_hash_new = dict()
    SA_hash_new = mdict.create("i32:i32")
    if len(SA_area) == 0:
        SA = current_list
        return SA,SA_hash

    previous_1 = -1
    previous_2 = -1

    next_1 = -1
    next_2 = -1
    j = 0

    len_current_list = len(current_list)
    while j < len_current_list:

        #item = SA_hash[current_list[j]]
        item = SA_hash[current_list[j]]

        if item != None:

            #print(item)
            next_2 = item
            next_1 = j

            for k in range(previous_2 + 1,next_2):
                SA.append(SA_area[k])
                SA_hash_new[SA_area[k]] = len(SA)-1

            #SA = SA + current_list[previous_1 + 1:next_1]
            for k in range(previous_1 + 1,next_1):
                SA.append(current_list[k])
                SA_hash_new[current_list[k]] = len(SA)-1

            SA.append(SA_area[item])
            SA_hash_new[SA_area[item]] = len(SA)-1

            previous_1 = j
            previous_2 = item

        j = j + 1

    j_1 = previous_1 + 1
    j_2 = previous_2 + 1

    #SA = SA + current_list[j_1:]
    for k in range(j_1, len(current_list)):
        SA.append(current_list[k])
        SA_hash_new[current_list[k]] = len(SA) - 1

    #SA = SA + SA_area[j_2:]
    for k in range(j_2, len(SA_area)):
        SA.append(SA_area[k])
        SA_hash_new[SA_area[k]] = len(SA) - 1

    return SA,SA_hash_new


##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--word', dest='word', action='store', default='aaabcaabcadcaabca')

    args = parser.parse_args()
    w = args.word

    SA_w = sorting_suffixes_via_icfl(w)

    print(SA_w)
