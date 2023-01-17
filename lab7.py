terminal_states = {'211111': 0, '22111': 1, '2221': 0}

successors = {
	'7': ['61', '52', '43'],
	'61' : ['511', '421'],
        '52' : ['421','322'],     '43' : ['421','331'],
        '511' : ['4111','3211'],  '421' : ['3211'],
        '322' : ['2221'],         '331' : ['3211'],
        '4111' : ['31111'],       '3211' : ['22111'],
        '31111' : ['211111'],
	# this dictionary is completed 
}

def util_value(state, agent):
        if state in terminal_states:
                return(terminal_states[state])

        if agent == 'MAX':
                return max_value(state)

        if agent == 'MIN':
                return min_value(state)
        
	#pass

def min_value(state):
        v=1000000000
        for each in successors[state]:
                v = min(v, util_value(each, 'MAX'))
        return v
                
	#pass

def max_value(state):
        v=-1000000000
        for each in successors[state]:
                v = max(v, util_value(each, 'MIN'))
        return v
	#pass

if __name__ == "__main__":
	print(util_value('7', 'MIN'))
	# OUTPUT: 1
