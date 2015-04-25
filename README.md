## Dependencies

These scripts need python 2.7+ and the following libraries to work:

1. pillow(~2.8.1)
2. numpy(~1.9.0)
3. python-opencv(~2.4.11)

The simplest way to install all of them is to install [python(x,y)](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2).

If you can't install python(x,y), You can install python, numpy and python-opencv seperately, then install pip and pillow.

1. Install python. Just use the installer from [python's website](https://www.python.org/downloads/)
2. Install numpy. Just use the installer from [scipy's website](http://www.scipy.org/scipylib/download.html). (You don't need scipy to run this project, so you can just install numpy alone).
3. Install python-opencv. Download the release from [its sourceforge site](http://sourceforge.net/projects/opencvlibrary/files/). (Choose the release based on your operating system, then choose version 2.4.11). The executable is just an archive. Extract the files, then copy `cv2.pyd` to the `lib/site-packages` folder on your python installation path.
4. Install pip. Download [the script for installing pip](https://bootstrap.pypa.io/get-pip.py), open cmd, go to the path where the downloaded script resides, and run `python get-pip.py`
5. Install pillow. Run `pip install pillow`. 

If you are running the code under Linux and the scripts throw `AttributeError: __float__`, make sure your pillow has jpeg support (consult [Pillow's document](http://pillow.readthedocs.org/en/latest/installation.html) e.g. try:

```
sudo apt-get install libjpeg-dev
sudo pip uninstall pillow
sudo pip install pillow
```

If you have any problem installing the dependencies, contact the author.

## How to generate the results

Enter the `src` directory, run

```
python main.py

```

It will use images under `dataset` directory to produce the results. The results will show up in `result` directory. Intermediate results will be saved too.

To see how long the script will take to generate the results (without saving them or the intermediate images), run

```
python main.py -t
```

##Directory structure

```
.
├─ README.md
├─ doc
│   └── report.docx
├─ dataset (source images)
│   └── ...
├─ result (the results)
│   └── ...
└─ src (the python source code)
      ├── perspective.py (perspective correction module)
      └── main.py (generate the results for the report)
```

##About

* [Github repository](https://github.com/joyeecheung/perspective-correction)
* Author: Qiuyi Zhang
* Time: Apr. 2015