import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # raise NotImplementedError
        max_val = float('inf')
        best_model = None
        for index  in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(index )
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                p = index ** 2 + 2 * model.n_features * index - 1
                bic_score = -2 * logL + p * logN
                if bic_score < max_val:
                    max_val = bic_score
                    best_model = model
            except:
                continue
            if best_model == None:
                return self.base_model(self.n_constant)
            else:
                return best_model
        #return best_model if best_model else self.base_model(self.n_constant)



class SelectorDIC(ModelSelector): # change
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # raise NotImplementedError
        min_value = float("-inf")
        best_model = None
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(i)
                logL = model.score(self.X, self.lengths)
                total_other_logL = 0
                for word in self.words:
                    if word != self.this_word:
                        other_x, other_lengths = self.hwords[word]
                        total_other_logL += model.score(other_x, other_lengths)
                avg_logL = total_other_logL/(len(self.words)-1)
                dic_scor = logL - avg_logL
                if dic_scor > min_value:
                    best_model = model
                    min_value = dic_scor
            except:
                continue
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        mean_scores = []

        split_method = KFold()
        try:
            for index in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(index)
                # calculate mean scores
                fold_scores = []
                for _, test_idx in split_method.split(self.sequences):
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    fold_scores.append(model.score(test_X, test_length))
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        num_components = range(self.min_n_components, self.max_n_components + 1)
        states = num_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)        
