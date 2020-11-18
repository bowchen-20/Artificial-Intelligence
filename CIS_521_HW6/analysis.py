# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question6():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'
    if answerEpsilon != None and answerLearningRate != None:
        return answerEpsilon, answerLearningRate
    else:
        return 'NOT POSSIBLE'
############
# Feedback #
############


# Just an approximation is fine.
feedback_question_1 = 3

feedback_question_2 = """
The assignment was very interesting to work on, some of the 
APIs were a bit hard to use but it got slightly better after-
wards. Seeing the training result as well as the animation def
helped with visualization.
"""

feedback_question_3 = """
I was a bit confused when reading the instruction for 1 and 3
but it got better once I figured out what I was supposed to do.
"""

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
