# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # return self.q_values[(state, action)]

        if (state, action) not in self.q_values:

            self.q_values[(state, action)] = 0.0

            return self.q_values[(state, action)]

        else:

            return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        value = float('-inf')

        for action in self.getLegalActions(state):

            q = self.getQValue(state, action)

            if value < q:

                value = q

        if value == float('-inf'):

            return 0.0

        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        action_list = []

        max_value = self.computeValueFromQValues(state)

        for actions in self.getLegalActions(state):

            if self.getQValue(state, actions) == max_value:

                action_list.append(actions)

        return random.choice(action_list) if action_list else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action

        actions = self.getLegalActions(state)

        if actions:
            if util.flipCoin(self.epsilon):
                return random.choice(actions)
            else:
                return self.computeActionFromQValues(state)
        else:
            return None

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        result = reward + self.discount * self.computeValueFromQValues(nextState)

        val = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * result

        self.q_values[(state, action)] = val

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        self.weights = util.Counter()

    def getWeights(self):
        # print(self.weights)
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q = 0

        for i in self.featExtractor.getFeatures(state, action).keys():

            q += self.getWeights()[i] * self.featExtractor.getFeatures(state, action)[i]


        # print(000)
        # print(q)
        # print(000)

        return q


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # diff = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        #
        # for i in self.featExtractor.getFeatures(state, action).keys():
        #
        #     self.weights[i] = self.weights[i] + self.alpha * diff * self.featExtractor.getFeatures(state, action)[i]

        cur_q = self.discount * self.computeValueFromQValues(nextState)

        diff = reward + cur_q - self.getQValue(state, action)

        for key in self.featExtractor.getFeatures(state, action):
            self.weights[key] = self.getWeights()[key] + self.alpha * diff * \
                                 self.featExtractor.getFeatures(state, action)[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # for i in self.weights.items():
            #     print(i)
            pass
