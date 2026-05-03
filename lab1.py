class Exercise1():

    def biggest(self, list):
        maxv = max(list)
        index = list.index(maxv)
        return list.pop(index)
    
    def get_target(self):
        target = input("Target: ")
        return int(target)
    
    def find_num(self, target_backup):
        coins = input("Enter coin types: ")
        coins = coins.split()
        coin = []
        total_list = []
        for c in coins:
            coin.append(int(c))
        self.backup_types = coin.copy()
        for n in range (len(self.backup_types)):
            self.coin_types = coin.copy()
            total = 0
            target = target_backup
            for i in range(n):
                self.biggest(self.coin_types)
            for i in range(len(self.coin_types)):
                biggestval = self.biggest(self.coin_types)
                total += target // biggestval
                remainder = target % biggestval
                target = remainder
            total_list.append(total)
        print(min(total_list))
                 
    def run(self):
        total = 0
        target = self.get_target()
        self.find_num(target)

class Exercise2():

    def fibrecurse(self, n):
        if n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            return self.fibrecurse(n - 2) + self.fibrecurse(n - 1)

    def fibite(self, n): # 0, 1, 1, 2, 3, 5, 8
        prev2 = 0
        prev1 = 1
        if n == 1:
            current = prev2
        elif n == 2:
            current = prev1
        else:
            for i in range(n-2):
                current = prev2 + prev1
                prev2 = prev1
                prev1 = current
        return current

class Exercise3():

    def __init__(self, input):
        self.sequence = input

    def balanced(self):
        open = 0
        for char in self.sequence:
            if char == '(':
                open += 1
            elif char == ')':
                if open == 0:
                    print("Not balanced")
                    break
                else:
                    open -= 1
        if open == 0:
            print("Balanced")
        else:
            print("Not balanced")
    
ex1_1 = Exercise1() 
ex1_1.run()

print()
ex2 = Exercise2()
print(ex2.fibrecurse(5))
print(ex2.fibite(5))
print(ex2.fibrecurse(10))
print(ex2.fibite(10))

print()
ex3_1 = Exercise3("((x*2)+(x-2)) - a[3] / a[10]")
ex3_1.balanced()

ex3_2 = Exercise3("(((x*2)+(x-2)) - a[3] / a[10] ")
ex3_2.balanced()

# done 
