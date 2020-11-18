############################################################
# CIS 521: Homework 7
############################################################


student_name = "Bowen Chen"

############################################################
# Imports
import string
import random
import math


############################################################

# Include your imports here, if any are used.


############################################################
# Section 1: Markov Models
############################################################

def tokenize(text):
    result = []
    text_cleaned = text.strip().split()
    for i in text_cleaned:
        for j in i:
            if j in string.punctuation:
                result.append(" " + j + " ")
            else:
                result.append(j)
        result.append(" ")
    #     print("".join(result))
    return "".join(result).split()


def ngrams(n, tokens):
    result = []
    new_list = list((n - 1) * ["<START>"]) + tokens + list(["<END>"])
    for i in range(n - 1, len(new_list)):
        result.append((tuple(new_list[i - n + 1:i]), new_list[i]))
    return result


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.grams = []

    def update(self, sentence):
        tokens = tokenize(sentence)
        self.grams += ngrams(self.n, tokens)

    def prob(self, context, token):
        context_count = 0
        token_count = 0

        for i in self.grams:
            if i[0] == context:
                context_count += 1
                if i[1] == token:
                    token_count += 1

        if context_count == 0:
            return 0
        else:
            return float(token_count) / float(context_count)

    def random_token(self, context):

        r = random.random()

        T = set()

        for i in range(len(self.grams)):
            if self.grams[i][0] == context:
                T.add(self.grams[i][1])

        T = list(T)
        T.sort()
        # print("Length of t is:")
        # print(len(T))
        # print(T)

        if len(T) == 0:
            return " "
        # print("loop")
        for token in T:
            # print("r before is:")
            # print(r)
            r -= self.prob(context, token)
            # print("r is:")
            # print(r)

            # print(111111)
            # print(token)
            # print(self.prob(context, token))
            # print(111111)
            if r < 0:
                return token

    def random_text(self, token_count):

        context = []

        result = []

        for i in range(self.n - 1):
            context.append("<START>")

        for j in range(token_count):

            generated_tokens = self.random_token(tuple(context))

            # print("generated_tokens:")
            # print(generated_tokens)

            result.append(generated_tokens)

            if generated_tokens == "<END>":

                context = []

                for k in range(self.n - 1):

                    context.append("<START>")

            else:

                context.append(generated_tokens)

                # print("Context before:")
                # print(context)

                context = context[1:len(context)]

                # print("Context after:")
                # print(context)

        generated_text = " ".join(i for i in result)

        return generated_text

    def perplexity(self, sentence):

        inside_term = 0

        tokenized_text = tokenize(sentence)

        for context, tokens in ngrams(self.n, tokenized_text):
            inside_term += math.log(self.prob(context, tokens))

        inside_result = 1 / math.exp(inside_term)

        return math.pow(inside_result, (1 / (len(tokenized_text) + 1)))


def create_ngram_model(n, path):
    model = NgramModel(n)

    with open(path, 'r') as f:
        for line in f.readlines():
            model.update(line)

    return model


############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = 2

feedback_question_2 = """
I was stuck on the randomized tokens part for a little bit 
and then Halley pointed out that I should be using a set 
instead of a list. I was not paying attention to duplicates
when I first started.
"""

feedback_question_3 = """
The assignment is pretty cool in general, and it's just that
sometimes the instruction appears a bit abstract and it would
be great if it could be made more clear.
"""

# print(tokenize(" This is an example. "))
# m = NgramModel(1)
# m.update("a b c d")
# m.update("a b a b")
# print(m.prob((), "a"))
# m = NgramModel(2)
# m.update("a b c d")
# m.update("a b a b")
# print(m.prob(("<START>",), "a"))
# m = NgramModel(1)
# m.update("a b c d")
# m.update("a b a b")
# random.seed(4)
# print([m.random_token(())for i in range(25)])
# print(m.random_text(13))
# m = NgramModel(1)
# m.update("a b c d")
# m.update("a b a b")
# print(m.perplexity("a b"))

# m = NgramModel(3)
# m.update("a b c d")
# m.update("a b a b")
# random.seed(2)
# m.random_text(15)
# print(m.random_text(15))
