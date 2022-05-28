from novainstrumentation import smooth

def rem_low_pass(s, win):
    l_p = smooth(s, win)
    return s-l_p

def sm(s, win):
    return smooth(s, win)





