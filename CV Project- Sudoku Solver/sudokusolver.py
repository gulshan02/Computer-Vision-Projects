
def cross(A, B):
    output = []
    for a in A:
        for b in B:
            output.append(a+b)
    return output

cols   = '123456789'
rows     = 'ABCDEFGHI'
cell  = cross(rows, cols)
output1=[]
for a in cols:
    output1.append(cross(rows,a))
output2=[]
for a in rows:
    output2.append(cross(a,cols))
output3=[]
for a in ('ABC','DEF','GHI'):
    for b in ('123','456','789'):
        output3.append(cross(a,b))
outputList = (output1+output2+output3)
json={}
for a in cell:
    for b in outputList:
        if a in b:
            if a in json:
                json[a].append(b)
            else:
                json[a]=[b]
# print json
peer={}
for a in cell:
    ans = set(sum(json[a],[]))-set([a])
    peer[a] = ans

# Convert grid to a dict of possible solution, {square: cols}, or return False if a contradiction is detected.
# To start, every square can be any digit; then give_values solution from the grid.
def parse_grid(grid):
    solution = {}
    for s in cell:
        solution[s]=cols
    solution_1=sudoku_solution(grid)
    for a in solution_1.items():
        if a[1] in cols:
            give_values_value = give_values(solution, a[0], a[1]) 
            if not give_values_value:
            # It will fail if we cannot give_values d to square s.
                return False
    return solution

# Convert grid into a dict of {square: char} with '0' for empties.
def sudoku_solution(grid):
    # Converting grid string to a list
    chars = []
    for c in grid:
        if c in cols or c == '0':
            chars.append(c)
    # print chars, "\n\n", len(chars)
    if len(chars) == 81:
        return dict(zip(cell, chars))
    return False

# Eliminate all the other solution (except d) from solution[s] and propagate.
# Return solution, except return False if a contradiction is detected.
def give_values(solution, s, d):
    # print d,"sdadadadas",solution[s],"sadadasdasda\n"
    other_solution = solution[s].replace(d, '')
    # print solution[s]
    output=[]
    # We will eliminate a from a from value of s.
    for a in other_solution:
        output.append(delete(solution,s,a))
    # print "output", output
    for a in output:
        if not a:
            return False
    return solution

# Delete d from solution[s]; propagate when solution or places <= 2.
# Return solution, except return False if a contradiction is detected.
def delete(solution, s, d):
    # If not in solution[s], that means it has already been deleted
    if d not in solution[s]:
        return solution
    # If a square s is reduced to one value d2, then delete d2 from the peer.
    solution[s] = solution[s].replace(d,'')
    if len(solution[s]) == 0:
	   return False
    elif len(solution[s]) == 1:
        d2 = solution[s]
        output=[]
        for a in peer[s]:
            output.append(delete(solution, a, d2))
        for a in output:
            if not a:
                return False
    # If a unit u is reduced to only one place for a value d, then put it there.
    for u in json[s]:
    	dplaces = []
        for a in u:
            if d in solution[a]:
                dplaces.append(a)
        # print dplaces
    	if len(dplaces) == 0:
    	    return False
    	elif len(dplaces) == 1:
    	    # d can only be in one place in unit, give_values it there
                if not give_values(solution, dplaces[0], d):
                    return False
        # print solution
    return solution

# This helps us to arrange the output in 2D matrix form.
def arrange(solution):
	if solution == False or solution == True:
		return []
	result= []
	temp=[]
	for r in rows:
		for c in cols:
			temp.append(int(solution[r+c]))
		result.append(temp)
		temp=[]
	return result

# We use depth-first search and propogation to find all possible solutions
def search(solution):
    if solution is False:
        return False 
    # If the following condition is True, that means sudoku is solved
    output=[]
    for a in cell:
        bool_value = len(solution[a]) ==1
        output.append(bool_value)
    # return solution
    # print output,"sadadadasdadasd\n"
    flag=0
    for a in output:
        if not a:
            flag=1
            break
    if flag==0:
        return solution
    # Chose the unfilled square s with the fewest possibilities.
    output=[]
    # Creating a list of all the unfilled cell.
    for a in cell:
        if len(solution[a]) > 1:
            output.append((len(solution[a]),a))
    # This is to find the unfilled cell with fewest possibilities.
    minimum = min(output)
    output = []
    for a in solution[minimum[1]]:
        given_values = give_values(solution.copy(), minimum[1], a)
        # print given_values
        output.append(search(given_values))
    # print output
    for a in output:
        if a:
            return a
    return False

def run(sudoku):
    grid=""
    for i in sudoku:
        for j in i:
            grid+=str(j)
    solution = parse_grid(grid)
    # print parse_grid(grid)
    solution = search(solution)
    # print("**************************************************************************")
    # print(solution)
    return arrange(solution)