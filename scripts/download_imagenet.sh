mkdir -p data/imagenet/train
mkdir -p data/imagenet/val
mkdir -p data/imagenet/test

# Download the images. This will take a while.
wget # TODO: add here download link for training set
tar -xvf ILSVRC2012_img_train.tar -C data/imagenet/train
rm data/imagenet/train/ILSVRC2012_img_train.tar
cd data/imagenet/train

find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

cd ../val

wget # TODO: add here download link for training set
tar -xvf ILSVRC2012_img_val.tar
rm ILSVRC2012_img_val.tar

cd ../test
wget # TODO: add here download link for training set
tar -xvf ILSVRC2012_img_test_v10102019.tar
rm ILSVRC2012_img_test_v10102019.tar