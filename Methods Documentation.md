# Methods Documentation

## getImage

Reads image file and returns it in grayscale and binarized.

#### Parameters

* file_path: string, default=""(file chooser will open in this case)
* blockSize: integer, default=505(must be odd)
* C: integer, default=-2

#### Output

* 2d image array

#### Library dependencies

* cv2
* tkinter

#### Example

x=getImage("dummy.jpg",21,1)



## noiseRemoval

Removes noise in a 2d image array (using cv2 MORPH_OPEN and MORPH_CLOSE).

#### Parameters

* gray: 2d array
* openIter: integer, default=3
* closeIter: integer, default=3

#### Output

* 2d image array

#### Library dependencies

* cv2

#### Example

x=cv2.imread("foobar.jpg")

x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)

y=noiseRemoval(x,closeIter=4)



## getCoverage

Reads 2d image array and returns its coverage

#### Parameters

* clean: 2d array

#### Output

* float

#### Library dependencies

* numpy

#### Example

x=cv2.imread("foobar.jpg")

x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)

y=getCoverage(x)



## get_img_from_fig

copied from JUN_NETWORKS at https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array

Returns 5d array from matplotlib plot.

#### Parameters

* fig: matplotlib figure

* dpi: integer, default=180

#### Output

* 5d image array(rgb)

#### Library dependencies

* io
* matplotlib
* numpy

#### Example

fig=matplotlib.pyplot.figure()

array=get_img_from_fig(fig)



## getRegions

Reads 2d image array and returns number of distinct regions found, as well as colored image array with regions boxed out.

#### Parameters

* sourceImage: 5d image array

* clean: 2d image array

* blockSize: integer, default=403(must be odd)

* C: integer, default=-4
* minArea: float, default=0.00025
* boxColor: string, default="red"

#### Output

* int
* 5d image array(rgb)

#### Library dependencies

* cv2
* matplotlib

#### Example

x=cv2.imread("foobar.jpg")

x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)

y=getImage("foobar.jpg")

a,b=getRegions(x,y,minArea=0.0004,boxColor="blue")



## processImage

Wrapper function for other functions, reads a file path and returns coverage, region count and regions boxed out image.

#### Parameters

* file_path: string, default=""(file chooser will open in this case)

* blockSize1: integer, default=505(must be odd)
* C1: integer, default=-2

* blockSize2: integer, default=403(must be odd)
* openIter: integer, default=3
* closeIter: integer, default-3

* C2: integer, default=-4
* minArea: float, default=0.00025
* boxColor: string, default="red"

#### Output

* float
* int
* 5d image array(rgb)

#### Library dependencies

* cv2
* matplotlib
* numpy
* tkinter

#### Example

a,b,c=processImage("foobar.jpg",blockSize2=333,minArea=0.0001)