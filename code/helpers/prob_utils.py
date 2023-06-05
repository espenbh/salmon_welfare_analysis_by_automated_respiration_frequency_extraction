# https://en.wikipedia.org/wiki/Binomial_distribution
# https://math.stackexchange.com/questions/4546935/i-throw-a-pot-of-10-dice-what-is-the-probability-that-i-get-at-least-two-sixe
import numpy as np

def comb(n,k):
    return np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))

#from numpy.math import factorial
def bin_prob(draws, succ, prob):
    return comb(draws, succ)*np.power(prob, succ)*np.power(1-prob,int(draws-succ))

def prob_random_clsf(max_cls_cnt, frame_num, prob = 1/14):
    random_clsf_prob = 0
    for i in range(max_cls_cnt, frame_num):
        random_clsf_prob = random_clsf_prob + bin_prob(frame_num, i, prob)
    return random_clsf_prob
