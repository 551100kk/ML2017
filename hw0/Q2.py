import sys
from PIL import Image

png1 = Image.open(sys.argv[1])
png2 = Image.open(sys.argv[2])
w, h = png1.size
ans = Image.new('RGBA', (w, h))
for i in range(w):
	for j in range(h):
		a = png1.getpixel((i,j))
		b = png2.getpixel((i,j))
		if a == b:
			ans.putpixel((i,j), (0, 0, 0, 0))
		else:
			ans.putpixel((i,j), b)
ans.save('ans_two.png')