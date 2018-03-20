MODEL="imagenet-vgg-verydeep-19.mat"

if [ -e $MODEL ]; then
  echo "You have already downloaded the model '$MODEL'"
else
  wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P ./data
  echo "Model '$MODEL' downloaded successfully"
fi
