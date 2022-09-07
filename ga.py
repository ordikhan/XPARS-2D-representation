import numpy as np
import copy
from sklearn.metrics import roc_curve, auc
import time
from threading import Thread



def compute_auc_roc(label_list, risk_list):
    fpr, tpr, _ = roc_curve(label_list, risk_list)
    return auc(fpr, tpr)


class Member:
    def __init__(self, a, idx, hard_fitness):
        self.cost = None
        self.idx = idx
        self.hard_fitness = hard_fitness
        self.init_chromosomes(a)

        # t = Thread(target=self.cost_fun)
        # t.start()

    def init_chromosomes(self, a):
        mid = int((a[-1] - a[0]) / 4)+1
        # mid = 1
        self.chromosomes = np.sort(np.random.randint(max(1, a[0] - mid), min(a[-1] + mid, 100), np.shape(a)))

    def cost_fun(self):
        for i, j in enumerate(self.idx):
            if len(self.hard_fitness[j]) != 0:
                self.hard_fitness[j][:, 1] = self.chromosomes[i]
        label_list = []
        risk_list = []
        for l in self.hard_fitness:
            for j in l:
                label_list.append(j[0])
                risk_list.append(j[1])
        self.cost = compute_auc_roc(label_list, risk_list)

    def reset_cost(self):
        self.cost = None

    def get_cost(self):
        # get cost of member
        if self.cost is None:
            self.cost_fun()
        return self.cost


class GA:
    """ genetic algorithm for solve this problem """

    def __init__(self, splited_data, a, iteration, population_number, len_age, len_pre, mutation_rate, mutation_use):
        self.a = np.array(a).T
        self.splited_data = splited_data
        self.iterarion = iteration
        self.population_number = population_number
        self.mutation_use = mutation_use
        self.mutation_rate = mutation_rate
        self.best_costs = []
        self.hard_fitness = []
        self.idxs = [[i * len_pre + k for k in range(len_pre)] for i in
                     range(int(self.a.shape[0] * self.a.shape[-1] / len_pre))]

        self.get_hard_fitness_list()

    def get_hard_fitness_list(self, list_rows=None):
        a = self.a.reshape((-1))
        if list_rows is None:
            self.hard_fitness = []
            datas = self.splited_data
        else:
            datas = []
            temp_a = []
            for i in list_rows:
                datas.append(self.splited_data[i])
                temp_a.append(a[i])
            a = np.array(temp_a)

        for i, data in enumerate(datas):
            temp = []
            if len(data) != 0:
                for index, row in data.iterrows():
                    temp.append([row['label'], a[i]])
            if list_rows is None:
                self.hard_fitness.append(np.array(temp))
            else:
                self.hard_fitness[list_rows[i]] = np.array(temp)
        return

    def test(self, datas):
        a = self.a.reshape((-1))
        hard_fitness = []
        label_list = []
        risk_list = []
        for i, data in enumerate(datas):
            temp = []
            if len(data) != 0:
                for index, row in data.iterrows():
                    temp.append([row['label'], a[i]])

                hard_fitness.append(np.array(temp))
        for l in hard_fitness:
            for j in l:
                label_list.append(j[0])
                risk_list.append(j[1])
        cost = compute_auc_roc(label_list, risk_list)
        return cost

    def solve(self, idx):
        """ solve GA"""
        best_cost = 0
        best_solotion = None
        a = self.a.reshape((-1))[self.idxs[idx]]
        self.members = [Member(a, self.idxs[idx], self.hard_fitness) for i in range(self.population_number)]
        pr = Member(a, self.idxs[idx], self.hard_fitness)
        pr.chromosome = a
        self.members.append(pr)

        for ite in range(self.iterarion):
            cost_list = np.array([self.members[i].get_cost() for i in range(len(self.members))])
            mutation_result = self.crossover_mutation(cost_list)

            self.members = sorted(self.members, key=lambda member: member.get_cost(), reverse=True)
            mutation_result = sorted(mutation_result, key=lambda member: member.get_cost(), reverse=True)
            temp = mutation_result[:int(self.population_number * self.mutation_use)]
            self.members = temp + self.members[:self.population_number - len(temp)]

            self.members = sorted(self.members, key=lambda member: member.get_cost(), reverse=True)
            costs = np.array([self.members[i].get_cost() for i in range(len(self.members))])
            argmax_cost = costs.argmax()

            if costs[argmax_cost] >= best_cost:
                best_cost = copy.deepcopy(costs[argmax_cost])
                best_solotion = copy.deepcopy(self.members[argmax_cost].chromosomes)
            # print("best cost :", best_cost, " at iteration :", ite, "  for idx : ", idx)
            self.best_costs[-1].append(best_cost)
        return best_solotion, best_cost

    def crossover_mutation(self, cost_list):
        total_fitness = cost_list.sum()
        possibility_fitness = cost_list / total_fitness
        children = []
        count_while = 0
        while len(children) < self.population_number and count_while < (10 * self.population_number):
            count_while += 1
            random_choice = np.random.choice(self.members, 2, p=possibility_fitness)
            child_ = self.__crossover(random_choice[0], random_choice[1])
            if child_ is None:
                continue
            children.append(child_[0])
            children.append(child_[1])
        children = self.mutation(children)
        return children

    @staticmethod
    def __crossover(parent1, parent2):
        children = []
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.reset_cost()
        child2.reset_cost()

        chromosomes = []
        for i in range(1, len(child1.chromosomes)):
            temp1 = np.concatenate([child1.chromosomes[:i], child2.chromosomes[i:]])
            temp2 = np.concatenate([child2.chromosomes[:i], child1.chromosomes[i:]])
            if not all(temp1 == child1.chromosomes) and not all(temp1 == child2.chromosomes) \
                    and all(temp1 == sorted(temp1)):
                chromosomes.append(temp1)

            if not all(temp2 == child1.chromosomes) and not all(temp2 == child2.chromosomes) \
                    and all(temp2 == sorted(temp2)):
                chromosomes.append(temp2)
        try:
            random_choice_chromosomes = np.random.choice(len(chromosomes), 2, replace=False)
            child1.chromosomes = chromosomes[random_choice_chromosomes[0]]
            child2.chromosomes = chromosomes[random_choice_chromosomes[1]]
            children.append(child1)
            children.append(child2)
        except:
            return None
        for child in children:
            child.get_cost()
        return children

    def mutation(self, crossover_result):
        mutations = []
        for p in crossover_result:
            if np.random.random() < self.mutation_rate:
                p.reset_cost()
                idx = np.random.randint(0, len(p.chromosomes))
                # mid = int((p.chromosomes[-1] - p.chromosomes[0]) / 4)+1
                mid = 1

                level = np.random.randint(max(1, p.chromosomes[0] - mid), min(p.chromosomes[-1] + mid, 100))
                if level > p.chromosomes[idx]:
                    for i, j in enumerate(p.chromosomes[idx:]):
                        if j < level:
                            p.chromosomes[i + idx] = level
                elif level < p.chromosomes[idx]:
                    for i, j in enumerate(p.chromosomes[:idx + 1]):
                        if j > level:
                            p.chromosomes[i] = level
                p.get_cost()
            mutations.append(p)

        return mutations

    def update_a(self, best_sol, idx):
        a = self.a.reshape((-1))
        a[self.idxs[idx]] = best_sol
        self.a = a.reshape(self.a.shape)

    def run(self, iters=1):
        number = []
        costs = []
        indeces = []
        main_count = 0
        for i in range(len(self.idxs)):
            count = 0
            for j in self.idxs[i]:
                count += len(self.splited_data[j])
            if count != 0:
                main_count += 1
            indeces.append((i, count))
        indeces = sorted(indeces, key=lambda x: x[1], reverse=True)
        for _ in range(iters):
            for j, icount in enumerate(indeces):
                i, count = icount

                if count >= 1:
                    start = time.time()
                    print(j, "/", main_count, " ==> ", end='')
                    self.best_costs.append([])
                    best_solotion, best_cost = self.solve(i)
                    number.append(count)
                    costs.append(best_cost)
                    self.update_a(best_solotion, i)
                    self.get_hard_fitness_list(list_rows=self.idxs[i])
                    print("time : ", time.time() - start, "  best cost : ", best_cost, end='\n')
        return costs,number
