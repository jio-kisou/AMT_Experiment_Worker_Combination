import collections


def true_correct_correct_cost(choice_prob_list, q_num, correct_answer, choice_num):
    cost = 0.0
    for i in range(choice_num):
        for j in range(choice_num):
            for k in range(choice_num):
                answers = [correct_answer, correct_answer, i, j, k]
                c = collections.Counter(answers)
                majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
                ratio = float(majority_list.count(correct_answer)) / float(len(majority_list))
                each_cost = ratio * choice_prob_list[q_num][i] * choice_prob_list[q_num][j] * choice_prob_list[q_num][k]
                cost = cost + each_cost
    return cost


def true_correct_mis_cost(choice_prob_list, q_num, correct_answer, mis_answer, choice_num):
    cost = 0.0
    for i in range(choice_num):
        for j in range(choice_num):
            for k in range(choice_num):
                answers = [correct_answer, mis_answer, i, j, k]
                c = collections.Counter(answers)
                majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
                ratio = float(majority_list.count(correct_answer)) / float(len(majority_list))
                each_cost = ratio * choice_prob_list[q_num][i] * choice_prob_list[q_num][j] * choice_prob_list[q_num][k]
                cost = cost + each_cost
    return cost


def true_mis_dif_cost(choice_prob_list, q_num, correct_answer, mis_answer1, mis_answer2, choice_num):
    cost = 0.0
    for i in range(choice_num):
        for j in range(choice_num):
            for k in range(choice_num):
                answers = [mis_answer1, mis_answer2, i, j, k]
                c = collections.Counter(answers)
                majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
                ratio = float(majority_list.count(correct_answer)) / float(len(majority_list))
                each_cost = ratio * choice_prob_list[q_num][i] * choice_prob_list[q_num][j] * choice_prob_list[q_num][k]
                cost = cost + each_cost
    return cost


def true_mis_same_cost(choice_prob_list, q_num, correct_answer, mis_answer, choice_num):
    cost = 0.0
    for i in range(choice_num):
        for j in range(choice_num):
            for k in range(choice_num):
                answers = [mis_answer, mis_answer, i, j, k]
                c = collections.Counter(answers)
                majority_list = [x[0] for x in c.items() if x[1] == max(c.values())]
                ratio = float(majority_list.count(correct_answer)) / float(len(majority_list))
                each_cost = ratio * choice_prob_list[q_num][i] * choice_prob_list[q_num][j] * choice_prob_list[q_num][k]
                cost = cost + each_cost
    return cost
