from MRTA.test_alloc import test
from heterogeneous_EC.overlord import run_threads
import numpy as np
import sys
sys.path.append("~/MRTA/")

if __name__ == "__main__":
	best_allocation = np.array(test())
	print("Types in () are for toy_env")
	print("Number of agents of type car (plane) allotted: ", best_allocation[0][1])
	print("Number of agents of type ship (car) allotted: ", best_allocation[1][1])
	print("Number of agents of type drone (ship) allotted: ", best_allocation[2][1])
	no_agent1 = best_allocation[0][1]
	no_agent2 = best_allocation[1][1]
	no_agent3 = best_allocation[2][1]
	no_agents = no_agent1 + no_agent2 + no_agent3

	#Start map and stop map can be chosen to run this on different maps
	start_map = 16
	stop_map = 17
	num_threads = 1
	# The sensorfootprints are in the file ./heterogeneous_EC/saved data/agents/sensor_footprint.txt
	# So depending on the number of agents assigned this file and the number of agents spawned has to be modified
	# for car -> 5, ship -> 7, plane -> 10
	file = open('./heterogeneous_EC/saved data/agents/sensor_footprint.txt','w')
	file.truncate()
	for i in range(no_agent1):
		file.write("5 ")
	for i in range(no_agent2):
		file.write("7 ")
	for i in range(no_agent3):
		file.write("10 ")
	file.write("-1")  #Indicating end of all agents
	file.close()

	run_threads(start_map, stop_map, num_threads, no_agents)

