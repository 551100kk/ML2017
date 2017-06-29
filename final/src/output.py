import math

def modify(path_test, path_xgb, path_ans):

	with open(path_test, 'r') as fp0, open(path_xgb, 'r') as fp1, open(path_ans, 'w') as fp2:
		fp0.readline()
		line = fp1.readline()
		fp2.write(line)

		for line in fp1:
			arr = line.replace('\n','').split(',')
			date = float(fp0.readline().replace('\n','').split(',')[1].replace('-', ''))
			price = float(arr[-1])

			if date <= 20150731:
				price += 20000
			elif date > 20150731 and date <= 20150831:
				price -= 240000
			elif date > 20150831 and date <= 20150930:
				price -= 360000

			elif date > 20150930 and date <= 20151031:
				price -= 200000
			elif date > 20151031 and date <= 20151130:
				price -= 100000
			elif date > 20151130 and date <= 20151231:
				price -= 180000

			elif date > 20151231 and date <= 20160131:
				price -= 20000
			elif date > 20160131 and date <= 20160230:
				price += 20000

			elif date > 20160230 and date <= 20160330:
				price += 150000
			else:
				price += 110000

			arr[-1] = str(price)
			fp2.write((',').join(arr) + '\n')