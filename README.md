# Using the UI to convert tvips to blo or tiff

Prerequisite: you have the Anaconda distribution installed

## On Unix based systems

1 Run the script setup.sh

2 Run the script converter.sh

## On Windows:

1 Open anaconda prompt and navigate to the folder containing setup.sh

2

# Steps to convert tvips to .blo or list of .tiff (Command line)

## Since an update to recorder.py in utils, this may no longer work.

1 Open anaconda shell and navigate to the folder containing record_c.py

```
cd path/to/tvips/data
```

2 Run the commands

2.1 For making a .blo file

```
python record_c.py --rotator --vbfradius 40  --otype blo "D:/Users/Jiwon/CuAg_low_26_20190412/ASTAR_TVIPS/Dummy/rec_20190412_183600_000.tvips" "D:/Users/Jiwon/CuAg_low_26_20190412/ASTAR_TVIPS/Dummy/test.blo" --dimension 150x150 --binning=8 --linscale=0-1000 --hysteresis 5

```

2.2 For making a list of tiffs in a folder

```

python record_c.py --rotator --vbfradius 40  --otype Individual "D:/Users/Jiwon/CuAg_low_26_20190412/ASTAR_TVIPS/Dummy/rec_20190412_183600_000.tvips" "D:/Users/Jiwon/CuAg_low_26_20190412/ASTAR_TVIPS/Dummy/dummytest/" --dimension 150x150 --binning=8 --linscale=0-1000 --hysteresis 5

```

Remove --binning if you don't want it. Make sure that the paths use / when you copy them and not \.
