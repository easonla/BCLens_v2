__author__ = "yhhsu"
from func import GetData
from func import FindFom

def main():
	# read config file
	parameters, mask = GetData.read_input("fominput")
	#result = (bestmassx1fom,xfom,bestmassx1snap,bestmassx1phi,bestmassx1theta,bestmassx1psi,bestmassx1time,counter) 
	result = FindFom.FindFom(parameters,mask)
	# read data path
	#data = read_func.read_data(config_x)
	# find fom
	# plot
	#output = plot(data)
	# save
	#save(output)

if __name__ == "__main__":
	main()