__author__ = "yhhsu"
import functions.read_config	as read_config
#import function.read_func


def main():
	# read config file
	config_x = read_config.read()
	print config_x

	# read data path
	#data = read_func.read_data(config_x)
	# find fom
	# plot
	#output = plot(data)
	# save
	#save(output)

if __name__ == "__main__":
	main()