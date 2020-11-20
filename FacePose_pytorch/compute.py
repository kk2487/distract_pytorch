def get_num(point_dict, name, axis):
	num = point_dict.get(f'{name}')[axis]
	num = float(num)
	return num

def cross_point(line1, line2):  
	x1 = line1[0]  
	y1 = line1[1]
	x2 = line1[2]
	y2 = line1[3]

	x3 = line2[0]
	y3 = line2[1]
	x4 = line2[2]
	y4 = line2[3]

	k1 = (y2 - y1) * 1.0 / (x2 - x1) 
	b1 = y1 * 1.0 - x1 * k1 * 1.0  
	if (x4 - x3) == 0: 
		k2 = None
		b2 = 0
	else:
		k2 = (y4 - y3) * 1.0 / (x4 - x3)
		b2 = y3 * 1.0 - x3 * k2 * 1.0
	if k2 == None:
		x = x3
	else:
		x = (b2 - b1) * 1.0 / (k1 - k2)
	y = k1 * x * 1.0 + b1 * 1.0
	return [x, y]

def point_line(point,line):
	x1 = line[0]  
	y1 = line[1]
	x2 = line[2]
	y2 = line[3]

	x3 = point[0]
	y3 = point[1]

	k1 = (y2 - y1)*1.0 /(x2 -x1) 
	b1 = y1 *1.0 - x1 *k1 *1.0
	k2 = -1.0/k1
	b2 = y3 *1.0 -x3 * k2 *1.0
	x = (b2 - b1) * 1.0 /(k1 - k2)
	y = k1 * x *1.0 +b1 *1.0
	return [x,y]

def point_point(point_1,point_2):
	x1 = point_1[0]
	y1 = point_1[1]
	x2 = point_2[0]
	y2 = point_2[1]
	distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
	return distance
