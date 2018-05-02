__
Make sure the folder is on your desktop.
Add our folder with subdirectories to the Set Path in Matlab.
To run:
>> main('trevor.png')
It will take about ~2 minutes to run SIFT and return results,
and then ~ minutes to run Bag-of-Words.
It will store each picture w/ features and feature matches in the folder 'Result Images'.
__
If you want to test a different user face, change line 243 in Evil Dectection/bag_words_p4/common/do_svm_evaluation.m to the name of the image, and then run main with the same image name.
