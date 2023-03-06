import imageio.v3 as iio
import glob



path = 'envmap/lego_point'
png_list = sorted(glob.glob(path + "/*.png"))

images = list()
for png in png_list:
    images.append(iio.imread(png))

iio.imwrite(path + "/video.mp4", images, fps=10)
