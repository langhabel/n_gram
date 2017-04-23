import numpy as np


class N_Gram(object):

    def __init__(self, alphabet, escape='C'):
        self.escape = escape
        self._alphabet = alphabet
        self._counts = {}

    def update(self, sequence, include_shortened=False, include_earlier=False):
        # print("update({}, {})".format(sequence, include_subsequences))
        # recursively include subsequences
        if self.hash_seq(sequence) != self.hash_seq([]):
            if include_earlier:
                for end in range(0, len(sequence)-1):
                    self.update(sequence=sequence[:end], include_shortened=include_shortened, include_earlier=False)
            if include_shortened:
                for start in range(1, len(sequence)):
                    self.update(sequence=sequence[start:], include_shortened=False, include_earlier=False)
        # include sequence
        sequence = self.hash_seq(sequence)
        if sequence in self._counts:
            self._counts[sequence] += 1
        else:
            self._counts[sequence] = 1

    def hash_seq(self, sequence=None):
        """Return a hashable object representing the sequence"""
        if sequence is None:
            return self.hash_seq([])
        return tuple(sequence)

    def weight(self, event, context):
        """compute unnormalized probability for event"""
        # print("--> weight({}|{})".format(event, context))
        # print(self.hash_seq(context), self.hash_seq([]))
        if self.hash_seq(context) == self.hash_seq([]):
            # return something like the uniform distribution
            # for empty contexts
            return 1 / (
                len(self._alphabet) + 1 - self.t(self.hash_seq([]))
            )
        else:
            gamma = self.gamma(context)
            p = self.weight(event=event, context=context[1:])
            # print("<-- weight({}|{})".format(event, context))
            try:
                return gamma * (
                    self.count(event=event, context=context) /
                    self.count(context=context)
                ) + (1 - gamma) * p
            except ZeroDivisionError:
                # I don't know how to deal with the situation where the given context
                # was not observed before. It leads to division by zero but it's not
                # explained in the paper as far as I understand.
                raise UserWarning("Don't know how to solve that case?!")
                return (1 - gamma) * p

    def distribution(self, context):
        """compute normalized probability distribution over all possible events"""
        dist = np.empty(len(self._alphabet))
        for idx, event in enumerate(self._alphabet):
            # print("w({}|{}):".format(event, context))
            dist[idx] = self.weight(event, context)
            # print("    {}".format(dist[idx]))
        return dist / dist.sum()

    def gamma(self, context):
        return self.count(context=context) / (
            self.count(context=context) + (
                self.t(context) / self.d(context)
            )
        )

    def t(self, context):
        counts = {}
        for e in context:
            if e in counts:
                counts[e] += 1
            else:
                counts[e] = 1
        return len(counts)

    def d(self, context):
        if self.escape == 'A':
            return self.t(context)
        elif self.escape == 'B':
            return 1
        elif self.escape == 'C':
            return 1
        elif self.escape == 'D':
            return 2
        else:
            raise UserWarning("unknown escape strategy '{}'".
                              format(self.escape))

    def k(self):
        if self.escape == 'A':
            return 0
        elif self.escape == 'B':
            return -1
        elif self.escape == 'C':
            return 0
        elif self.escape == 'D':
            return -0.5
        else:
            raise UserWarning("unknown escape strategy '{}'".
                              format(self.escape))

    def count(self, event=None, context=None):
        if event is None:
            count_sum = 0
            for e in self._alphabet:
                count_sum += self.count(event=e, context=context)
            return count_sum
        else:
            sequence = self.hash_seq(np.append(context, event))
            if sequence in self._counts:
                return self._counts[sequence] + self.k()
            else:
                return 0

if __name__ == '__main__':
    songs = [
        [62, 69, 69, 71, 71, 69, 67, 66, 62, 69, 69, 71, 71, 69, 66, 62, 69, 69, 71, 71, 69, 67, 66, 74, 74, 69, 66, 74, 74, 69, 66],
        [55, 60, 60, 60, 60, 57, 55, 57, 59, 60, 55, 60, 60, 60, 60, 60, 58, 57, 55, 57, 57, 57, 55, 53, 52, 57, 60, 59, 57, 56, 57],
        [65, 65, 69, 72, 74, 72, 69, 72, 70, 69, 67, 65, 65, 65, 69, 72, 74, 72, 69, 72, 70, 69, 67, 65, 72, 72, 70, 69, 72, 70, 69, 72, 72, 74, 74, 72, 70, 69, 72, 72, 72, 69, 77, 72, 72, 72, 69, 72, 72, 65],
        [62, 69, 69, 71, 71, 69, 67, 66, 62, 69, 69, 71, 71, 69, 66, 62, 69, 69, 71, 71, 69, 67, 66, 62, 69, 69, 71, 71, 69, 66]
        ]
    alphabet = [52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    event_idx = {}
    for idx, event in enumerate(alphabet):
        event_idx[event] = idx
    best_entropy = 2.75
    corpus_cross_entropy = 0
    n_events_in_corpus = 0
    for song in songs:
        song = np.array(song)
        song_cross_entropy = 0
        n_gram = N_Gram(alphabet=alphabet)
        for time in range(1, len(song)):
            context = song[0:time-1]
            print(context)
            n_gram.update(context, include_shortened=True, include_earlier=False)
            dist = n_gram.distribution(context)
            song_cross_entropy -= np.log2(dist[event_idx[song[time]]])
        corpus_cross_entropy += song_cross_entropy
        song_cross_entropy /= len(song)
        n_events_in_corpus += len(song)
        print("song cross entropy:", song_cross_entropy)
    corpus_cross_entropy /= n_events_in_corpus
    print("corpus cross entropy:", corpus_cross_entropy)