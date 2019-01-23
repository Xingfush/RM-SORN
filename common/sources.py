from __future__ import division
# from pylab import *
import numpy as np
import itertools
import utils
import common.synapses as synapses

class AbstractSource():
    def __init__(self):
        """Initialize all relevant variables."""
        raise NotImplementedError
    def next(self):
        """Returns the next input"""
        raise NotImplementedError
    def global_range(self):
        """Returns the maximal global index of all inputs"""
        raise NotImplementedError
    def global_index(self):
        """
        TODO check if this is really the case!
        Returns the current global(unique) index of the current input "character"
        :return:
        """
        raise NotImplementedError
    def generate_connection_e(self,N_e):
        """Generate connection matrix W_eu from input to the excitatory population"""
        raise NotImplementedError
    def generate_connection_i(self,N_i):
        """Generate connection matrix W_iu from input to the inhibitory population"""
        raise NotImplementedError

class CountingSource(AbstractSource):
    """
    Source for the counting task.
    Different of words are presented with individual probabilities.
    """
    def __init__(self,words,probs,N_u_e,N_u_i,avoid=False):
        """
        Initialize variables.
        :param words: list, The words to present alternatively.
        :param probs: matrix, The probabilities of transitioning between word i and j.
        :param N_u_e: int, number of active units per step
        :param N_u_i: int, number of active inhibitory units per step
        :param avoid: bool, avoid same excitatory units for different words
        """
        self.word_index = 0
        self.ind = 0
        self.words = words
        self.probs = probs
        self.N_u_e = int(N_u_e)
        self.N_u_i = int(N_u_i)
        self.avoid = avoid
        self.alphabet = np.unique(list("".join(words)))
        self.N_a = len(self.alphabet)
        self.lookup = dict(zip(self.alphabet,range(self.N_a)))
        self.glob_ind = [0]
        self.glob_ind.extend(np.cumsum(list(map(len,words))))
        self.predict = self.predictability()
        self.reset()

    @classmethod
    def init_simple(cls,N_words,N_letters,word_length,max_fold_prob,
                    N_u_e,N_u_i,avoiding,words=None):
        """
        Construct the arguments for the source to make it usable for the
        cluster

        Parameters:
            N_words: int
                Number of different words
            N_letters: int
                Number of letters to generate words from
            word_length: list
                Range of length (unimodal distribution)
            max_fold_prob: float
                maximal probability difference between words
            N_u_e: int
                Number of active excitatory units per step
            N_u_i: int
                Number of active inhibitory units per step
            avoid: bool
                Avoid same excitatory units for different words
        """
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        assert(N_letters <= len(letters))
        letters = np.array([x for x in letters[:N_letters]])

        if words is None:
            words = []
            for i in range(N_words):
                word = letters[np.random.randint(0,N_letters,np.random.randint(word_length[0],
                                                                                word_length[1]+1))]
                words.append(''.join(word))
        else:
            assert (N_words==len(words) and
                    N_letters==len(np.unique(''.join(words))))
        probs = np.array([np.random.rand(N_words)*(max_fold_prob-1)+1]*N_words)
        # Normalize the transition probs.
        probs /= sum(probs,1)

        return CountingSource(words,probs,N_u_e,N_u_i,avoid=avoiding)

    def generate_connection_e(self,N_e):
        W = np.zeros((N_e,self.N_a))
        available = set(range(N_e))
        for a in range(self.N_a):
            temp = np.random.choice(list(available),self.N_u_e)
            W[temp,a]=1
            if self.avoid:
                available = available.difference(temp)
        if '_' in self.lookup:
            W[:,self.lookup['_']] = 0

        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        ans = synapses.create_matrix((N_e,self.N_a),c)
        ans.W = W
        return ans

    def generate_connection_i(self,N_i):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connection=False)
        ans = synapses.create_matrix((N_i,self.N_a),c)
        W = np.zeros((N_i,self.N_a))
        if N_i>0:
            available = set(range(N_i))
            for a in range(self.N_a):
                temp = np.random.choice(list(available),self.N_u_i)
                W[temp,a] = 1
            if '_' in self.lookup:
                W[:,self.lookup['_']] = 0
        ans.W = W
        return ans

    def char(self):
        """
        Return the current char that is pointed by word_index and index.
        :return: char
        """
        word = self.words[self.word_index]
        return word[self.ind]

    def index(self):
        """
        Return the index of current char in lookup dictionary.
        :return:
        """
        character = self.char()

        # import ipdb; ipdb.set_trace()
        ind = self.lookup[character]
        return ind

    def next_word(self):
        """
        Helper method for the next bellow.
        Notice the usage of np.nonzero, unwrapping via two [].
        """
        self.ind = 0
        w = self.word_index
        p = self.probs[w,:]
        self.word_index = np.nonzero(np.random.rand()<=np.cumsum(p))[0][0]

    def next(self):
        """
        Getting the next char and return its vector representation.
        Consistent with the W_eu and W_iu connection matrix.
        :return: 1-D array, (N_a,), only 1 in array indicating the occurrence.
        """
        self.ind = self.ind+1
        string = self.words[self.word_index]
        if self.ind >=len(string):
            self.next_word()
        ans = np.zeros(self.N_a)
        ans[self.index()] = 1
        return ans

    def current(self):
        """
        Getting the current char and return its vector representation.
        Design for RMSorn class and feedback tasks.
        :return: 1-D array.
        """
        ans = np.zeros(self.N_a)
        ans[self.index()] = 1
        return ans

    def reset(self):
        self.next_word()
        self.ind = -1

    def global_index(self):
        """Global denotes the index in list consisting of all words in char"""
        return self.glob_ind[self.word_index]+self.ind

    def global_range(self):
        """The sum of length of all words."""
        return self.glob_ind[-1]

    def trial_finished(self):
        """If the testing of current word is finished."""
        return self.ind+1>=len(self.words[self.word_index])

    def predictability(self):
        temp = self.probs
        for n in range(10):
            temp = temp.dot(temp)
        final = temp[0,:]
        probs = list(map(len,self.words))
        probs = np.array(probs)
        probs = (probs + self.probs.max(1)-1)/probs
        return sum(final*probs)

class TrialSource(AbstractSource):
    """
    This source takes any other source and gives it a trial-like
    structure with blank periods inbetween stimulation periods
    The source has to implement a trial_finished method that is True
    if it is at the end of one trial
    """
    def __init__(self,source,blank_min_length,blank_var_length,
                 defaultstim,resetter=None):
        assert(hasattr(source,'trial_finished'))
        self.source = source
        self.blank_min_length = blank_min_length
        self.blank_var_length = blank_var_length
        self.reset_blank_length()
        self.defaultstim = defaultstim
        self.resetter = resetter
        self._reset_source()
        self.blank_step = 0

    def reset_blank_length(self):
        if self.blank_var_length > 0:
            self.blank_length = self.blank_min_length\
                                + np.random.randint(self.blank_var_length)
        else:
            self.blank_length = self.blank_min_length

    def next(self):
        if not self.source.trial_finished():
            return self.source.next()
        else:
            if self.blank_step >= self.blank_length:
                self.blank_step = 0
                self._reset_source()
                self.reset_blank_length()
                return self.source.next()
            else:
                self.blank_step +=1
                return self.defaultstim

    def _reset_source(self):
        if self.resetter is not None:
            getattr(self.source,self.resetter)()

    def global_range(self):
        return self.source.global_range()

    def global_index(self):
        if self.blank_step>0:
            return -1
        return self.source.global_index()

    def generate_connection_e(self,N_e):
        return self.source.generate_connection_e(N_e)

    def generate_connection_i(self,N_i):
        return self.source.generate_connection_i(N_i)

class NoSource(AbstractSource):
    """
    No input for the spontaneous conditions.
    """
    def __init__(self,N_i=1):
        self.N_i = N_i
    def next(self):
        return np.zeros((self.N_i))
    def global_range(self):
        return 1
    def global_index(self):
        return -1

    def generate_connection_e(self,N_e):
        c = utils.Bunch(lamb=np.inf,
                        avoid_self_connections=False)
        tmpsyn = synapses.create_matrix((N_e,self.N_i),c)
        tmpsyn.set_synapses(tmpsyn.get_synapses()*0)
        return tmpsyn

    def generate_connection_i(self,N_i):
        c = utils.Bunch(lamb=np.inf,
                        avoid_self_connections=False)
        return synapses.create_matrix((N_i,self.N_i),c)

