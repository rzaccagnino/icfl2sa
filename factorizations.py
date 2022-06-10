import os
import time

ALPHABET = [chr(i) for i in range(512)]
#ALPHABET = ['a','c','g','t']
INVERSE_ALPHABET = ALPHABET[::-1]

def index_in_alphabet(t,typ_alphabet_list):
    i = typ_alphabet_list.index(t)
    return typ_alphabet_list.index(t)


# Given a list of factors return the fingerprint
def compute_fingerprint_by_list_factors(list_fact):

    #print(list_fact)

    interval_sequence = []

    txt = ''
    prev_factor = list_fact[0]

    if len(list_fact) == 1:
        interval_sequence.insert(0,len(prev_factor))

    if prev_factor.endswith("\n"):
        prev_factor = prev_factor[0:len(prev_factor) - 1]

        if len(list_fact) == 1:
            interval_sequence[0] = len(prev_factor)
            txt = txt + str(len(prev_factor)) + "\n\n"
            txt = txt + "[" + str(prev_factor) + "]\n"

        if prev_factor == "<<" or prev_factor == ">>":
            if len(list_fact) > 1:
                txt = txt + prev_factor + ","
            else:
                txt = txt + prev_factor

            #print(txt)

        len_fact = len(prev_factor)
        count_equal_factors = 1

        for i in range(1, len(list_fact)):

            next_factor = list_fact[i]

            if next_factor.endswith("\n"):
                next_factor = next_factor[0:len(next_factor) - 1]

            if next_factor == "<<" or next_factor == ">>":

                if prev_factor == "<<" or prev_factor == ">>":
                    txt = txt + next_factor
                else:
                    txt = txt + str(len_fact) + "," + str(count_equal_factors)
                print(txt)

                if i == len(list_fact) - 1:
                    txt = txt + "," + next_factor
                    print(txt)
                else:
                    if prev_factor == "<<" or prev_factor == ">>":
                        txt = txt + ","
                    else:
                        txt = txt + "," + next_factor + ","

                    prev_factor = next_factor
                    count_equal_factors = 1
                    print(txt)

            else:

                # Devo stampare quello trovato fino ad ora
                if prev_factor == "<<" or prev_factor == ">>":
                    if i == len(list_fact) - 1:
                        txt = txt + str(len(next_factor))

                        prev_factor = next_factor
                        len_fact = len(prev_factor)
                    else:
                        txt = txt + str(len_fact)
                        print(txt)

                        if i < len(list_fact) - 1:
                            txt = txt + ","
                            print(txt)

                            prev_factor = next_factor
                            len_fact = len(prev_factor)

                            count_equal_factors = 1
                        else:
                            txt = txt + "," + str(len(next_factor))

            txt = txt + "\n"
            txt = txt + str(list_fact) + "\n\n"
            print(txt)

    txt.replace("<<", "0")
    txt.replace(">>", "-1")
    print(txt)
    print(str(interval_sequence))


# ------------------------ CFL ---------------------------------------------------------------------
# CFL - Lyndon factorization - Duval's algorithm
def CFL(word):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] < word[i - 1]:
                while k < i:
                    # print(word[k:k + j - i])
                    CFL_list.append(word[k:k + j - i])
                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] > word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list


def contains(window, preview, next_c):
    """Check if [window + preview] contains [preview + next_c]"""
    return ''.join(preview + [next_c]) in ''.join(window + preview)


def find_index(window, preview):
    """Return the first index of preview occurring in window
       relative to the window size"""
    dictionary = ''.join(window + preview)
    return len(window) - dictionary.find(''.join(preview))



# ------------------------ CFL IN ---------------------------------------------------------------------
# CFL IN - Lyndon factorization on inverse alphabet
def CFL_inverse(word):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] > word[i - 1]:

                f = ''
                while k < i:
                    # print(word[k:k + j - i])
                    f = f + word[k:k + j - i]
                    k = k + j - i
                CFL_list.append(f)
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] < word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list


# ----------------------- ICFL ----------------------------------------------------------------------
# ICFL recursive (without using of compute_br)- Inverse Lyndon factorization
def ICFL_recursive(word):
    """In this version of ICFL, we don't execute compute_br - one only O(n) scanning of word"""
    br_list = []
    icfl_list = []

    compute_icfl_recursive(word, br_list, icfl_list)

    return icfl_list

def compute_icfl_recursive(word, br_list, icfl_list):

    # At each step compute the current bre
    pre_pair = find_pre(word)
    current_bre_quad = find_bre(pre_pair)
    br_list.append(current_bre_quad)

    #if current_bre_quad[1] == '' and current_bre_quad[0].find('$') >= 0:
    if current_bre_quad[1] == '' and current_bre_quad[0].find(chr(200)) >= 0:
        w = current_bre_quad[0]
        icfl_list.insert(0, w[:len(w) - 1])
        return
    else:
        compute_icfl_recursive(current_bre_quad[1] + current_bre_quad[2], br_list, icfl_list)
        if len(icfl_list[0]) > current_bre_quad[3]:
            icfl_list.insert(0, current_bre_quad[0])
        else:
            icfl_list[0] = current_bre_quad[0] + icfl_list[0]
        return


def find_pre(word):
    if len(word) == 1:
        #return (word + "$", '')
        return (word + chr(200), '')
    else:
        i = 0
        j = 1

        while j < len(word) and word[j] <= word[i]:
            if word[j] < word[i]:
                i = 0
            else:
                i = i + 1
            j = j + 1

        if j == len(word):
            #return (word + "$", '')
            return (word + chr(200), '')
        else:
            return (word[0:j + 1], word[j + 1:len(word)])


def find_pre_for_alphabet(word,list_alphabet):
    if len(word) == 1:
        #return (word + "$", '')
        return (word + chr(200), '')
    else:
        i = 0
        j = 1

        while j < len(word) and index_in_alphabet(word[j],list_alphabet) <= index_in_alphabet(word[i],list_alphabet):
            if index_in_alphabet(word[j],list_alphabet) < index_in_alphabet(word[i],list_alphabet):
                i = 0
            else:
                i = i + 1
            j = j + 1

        if j == len(word):
            #return (word + "$", '')
            return (word + chr(200), '')
        else:
            return (word[0:j + 1], word[j + 1:len(word)])


def find_bre(pre_pair):
    w = pre_pair[0]
    v = pre_pair[1]

    #if v == '' and w.find('$') >= 0:
    if v == '' and w.find(chr(200)) >= 0:
        return (w, '', '', 0)
    else:
        n = len(w) - 1

        f = border(w[:n])

        i = n
        last = f[i-1]

        while i > 0:
            if w[f[i-1]] < w[n]:
                last = f[i-1]
            i = f[i-1]

        return (w[:n - last], w[n - last:n + 1], v, last)


def border(p):
    l = len(p)
    pi = [0]
    k = 0
    for i in range(1, l):
        while (k > 0 and p[k] != p[i]):
            k = pi[k-1]
        if (p[k] == p[i]):
            pi.append(k +1)
            k = k + 1
        else:
            pi.append(k)

    return pi


def compute_br(w, br_list):
    pre_pair = find_pre(w)
    bre_quad = find_bre(pre_pair)

    if bre_quad[1] == '':
        br_list.append(bre_quad)
        return br_list
    else:
        br_list.append(bre_quad)
        compute_br(bre_quad[1] + bre_quad[2], br_list)

# ------------------------  FAST ICFL ---------------------------------------------------------------------
# FAST ICFL (RIGHT TO LEFT) - Inverse Lyndon factorization - steps:
#   1) CFL_in
#   2) scan the sequence of factors in the grouping to ensure the "border" property
def FAST_ICFL_RL(word):

    list_factors_cfl_in = CFL_inverse(word)

    list_factors_icfl = []
    if len(list_factors_cfl_in) == 1:
        return list_factors_cfl_in

    dec = 0
    prev_f = list_factors_cfl_in[len(list_factors_cfl_in) - 2]  # f_n
    current_f = list_factors_cfl_in[len(list_factors_cfl_in) - 1]  # f_{i-1}
    for i in range(len(list_factors_cfl_in)-1, -1, -1):

        i = i - dec
        dec = 0

        # check if current_f starts with next_f
        if prev_f.startswith(current_f):

            new_f = prev_f + current_f

            if len(list_factors_icfl) == 0:
                list_factors_icfl.insert(0, new_f)

                if i == 1:
                    return list_factors_icfl
                else:
                    #prev_f = list_factors_cfl_in[i-2]
                    #current_f = new_f
                    prev_f = list_factors_cfl_in[i-3]
                    current_f = list_factors_cfl_in[i-2]
                    dec = 1
            else:
                # check if a border of f_if_{i+1} exists which is prefix of f_{i+2}
                check_f = list_factors_icfl[0]

                # Proprietà generale
                # a border not exists?
                if not border_property(new_f, check_f):
                    #list_factors_icfl.append(new_f)

                    if i == 1:
                        list_factors_icfl.insert(0,new_f)
                        return list_factors_icfl
                    else:
                        prev_f = list_factors_cfl_in[i - 2]
                        current_f = new_f

                else:
                    list_factors_icfl.insert(0, current_f)

                    if i == 2:
                        list_factors_icfl.insert(0,prev_f)
                        return list_factors_icfl
                    else:
                        #prev_f = list_factors_cfl_in[i - 2]
                        #current_f = prev_f
                        prev_f = list_factors_cfl_in[i - 3]
                        current_f = list_factors_cfl_in[i - 2]
                        dec = 1


                '''
                if new_f.startswith(check_f) and new_f.endswith(check_f):
                    list_factors_icfl.insert(0, current_f)
                    current_f = prev_f
                else:
                    current_f = new_f
                    '''
        else:

            if i == 1:
                list_factors_icfl.insert(0, current_f)
                list_factors_icfl.insert(0,prev_f)
                return list_factors_icfl
            else:
                #if i == len(list_factors_cfl_in) -1:
                list_factors_icfl.insert(0, current_f)
                current_f = prev_f
                prev_f = list_factors_cfl_in[i-2]


    return list_factors_icfl


# FAST ICFL (LEFT TO RIGHT)
def FAST_ICFL_LR(word):

    list_factors_cfl_in = CFL_inverse(word)

    list_factors_icfl = []
    if len(list_factors_cfl_in) == 1:
        return list_factors_cfl_in

    current_f = list_factors_cfl_in[0]
    next_f = list_factors_cfl_in[1]
    for i in range(1, len(list_factors_cfl_in), 1):

        if current_f.startswith(next_f):

            new_f = current_f + next_f

            if i == len(list_factors_cfl_in) - 1:
                list_factors_icfl.append(new_f)
                return list_factors_icfl
            else:
                check_f = list_factors_cfl_in[i+1]

                # a border not exists?
                if not border_property(new_f, check_f):

                    if i == len(list_factors_cfl_in) - 2:
                        list_factors_icfl.append(new_f)
                        list_factors_icfl.append(check_f)

                        return list_factors_icfl
                    else:
                        current_f = new_f
                        next_f = check_f

                else:
                    '''
                    if (i+1) == len(list_factors_cfl_in) - 1:
                        new_f = new_f + check_f
                        list_factors_icfl.append(new_f)
                        return list_factors_icfl
                    else:
                        print('daje2')
                        list_factors_icfl.append(current_f)

                        next_f = list_factors_cfl_in[i + 2]
                        current_f = list_factors_cfl_in[i +1]
                        inc = 1
                    '''
                    new_f = new_f + check_f
                    list_factors_icfl.append(new_f)
                    return list_factors_icfl
        else:

            if i == len(list_factors_cfl_in) - 1:
                list_factors_icfl.append(current_f)
                list_factors_icfl.append(next_f)
                return list_factors_icfl
            else:
                list_factors_icfl.append(current_f)
                current_f = next_f
                next_f = list_factors_cfl_in[i+1]


    return list_factors_icfl



# if new_f "has a border which is prefix of" check_f
def border_property(new_f, check_f):

    for i in range(1, len(check_f)+1):
        prefix = check_f[:i]

        if new_f.startswith(prefix) and new_f.endswith(prefix):
            return True

    return False



# ------------------------ CFL_icfl ---------------------------------------------------------------------
# CFL factorization - ICFL subdecomposition
def CFL_icfl(word,C):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] < word[i - 1]:
                while k < i:
                    # print(word[k:k + j - i])
                    w = word[k:k + j - i]
                    if len(w) <= C:
                        CFL_list.append(word[k:k + j - i])
                    else:
                        ICFL_list_recursive = ICFL_recursive(w)

                        # Insert << to indicate the begin of the subdecomposition of w
                        CFL_list.append("<<")
                        for v in ICFL_list_recursive:
                            CFL_list.append(v)
                        # Insert >> to indicate the end of the subdecomposition of w
                        CFL_list.append(">>")

                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] > word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list

# CFL - Lyndon factorization - Duval's algorithm - on a specific algorithm
def CFL_for_alphabet(word, list_alphabet):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or index_in_alphabet(word[j - 1],list_alphabet) < index_in_alphabet(word[i - 1],list_alphabet):
                while k < i:
                    # print(word[k:k + j - i])
                    CFL_list.append(word[k:k + j - i])
                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if index_in_alphabet(word[j - 1],list_alphabet) > index_in_alphabet(word[i - 1],list_alphabet):
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list
# ---------------------------------------------------------------------------------------------

def ICFL_recursive_for_alphabet(word,list_alphabet):
    """In this version of ICFL, we don't execute compute_br - one only O(n) scanning of word"""
    br_list = []
    icfl_list = []

    compute_icfl_recursive_for_alphabet(word, br_list, icfl_list, list_alphabet)

    return icfl_list

# ICFL recursive (without using of compute_br)- Inverse Lyndon factorization - on a specific algorithm
def compute_icfl_recursive_for_alphabet(word, br_list, icfl_list, list_alphabet):

    # At each step compute the current bre
    pre_pair = find_pre_for_alphabet(word,list_alphabet)
    current_bre_quad = find_bre_for_alphabet(pre_pair,list_alphabet)
    br_list.append(current_bre_quad)

    if current_bre_quad[1] == '' and current_bre_quad[0].find('$') >= 0:
        w = current_bre_quad[0]
        icfl_list.insert(0, w[:len(w) - 1])
        return
    else:
        compute_icfl_recursive(current_bre_quad[1] + current_bre_quad[2], br_list, icfl_list)
        if len(icfl_list[0]) > current_bre_quad[3]:
            icfl_list.insert(0, current_bre_quad[0])
        else:
            icfl_list[0] = current_bre_quad[0] + icfl_list[0]
        return


def find_bre_for_alphabet(pre_pair,list_alphabet):
    w = pre_pair[0]
    v = pre_pair[1]

    if v == '' and w.find('$') >= 0:
        # return (w[:len(w)-1], '', '', 0)
        return (w, '', '', 0)
    else:
        n = len(w) - 1

        f = border(w[:n])

        i = n
        last = f[i - 1]

        while i > 0:
            if index_in_alphabet(w[f[i - 1]],list_alphabet) < index_in_alphabet(w[n],list_alphabet):
                last = f[i - 1]
            i = f[i - 1]

        return (w[:n - last], w[n - last:n + 1], v, last)


# CHRONOMETHER FACTORIZATION ###########################################################################################
def chronomether(factorization):

    PATH = 'largecalgarycorpus/original_old'

    start = time.time()

    for _, _, files in os.walk(PATH):

        for f in files:
            f_path = os.path.join(PATH,f)

            with open(f_path, errors="ignore") as f1:
                text = f1.read()

                factorization(text)

    end = time.time()

    print(end - start)


if __name__ == "__main__":

    '''
    w = 'ccacdaaacadacadca'

    list_factors = FAST_ICFL_LR(w)
    print("fast icfl: {}".format(list_factors))

    list_factors_icfl = ICFL_recursive(w)
    print("icfl: {}".format(list_factors_icfl))
    '''

    w = 'bcabddbaaaabcbddd'
    list_factors_icfl = ICFL_recursive(w)
    print("icfl: {}".format(list_factors_icfl))
