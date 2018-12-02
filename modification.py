import csv

with open('Data/jester-modified.csv', 'w') as jm:
	writer = csv.writer(jm , lineterminator = '\r')
	# temp = ["user_id" , "item_id" , "rating"]
	# writer.writeheader(temp)
	with open('Data/jester-combined.csv', 'r') as jc:
		spamreader = csv.reader(jc, delimiter=',')
		count = 0
		for row in spamreader:
			u_id = str(count)
			c = 0
			for r in row[1:]:
				i_id = str(c)
				c+= 1
				r = float(r)
				if(-10 <= r  < -5 ):
					r = 1
				elif(-5 <= r < 0):
					r = 2
				elif(0 <= r < 5):
					r = 3
				elif(r > 5 and r!= 99):
					r = 4
				else:
					continue
				temp = [u_id , i_id , str(r)]

				writer.writerow(temp)
				

			count += 1
		
		# writer.writerow(row)
	jm.close()