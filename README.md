# Video Neural Style Transfer
This is an algorithm that implements neural style transfer for images and videos. 

Webpage: https://dliangsta.github.io/Video-Neural-Style-Transfer/ </br>
Results: https://github.com/dliangsta/Video-Neural-Style-Transfer/tree/master/results </br>

### Usage

```
usage: python3 src/main.py [-h] [--content_type CONTENT_TYPE]
               [--content_path CONTENT_PATH] [--style_path STYLE_PATH]
               [--vgg_path VGG_PATH] [--frame_input_path FRAME_INPUT_PATH]
               [--frame_output_path FRAME_OUTPUT_PATH]
               [--output_path OUTPUT_PATH]
               [--epoch_input_path EPOCH_INPUT_PATH]
               [--epoch_output_path EPOCH_OUTPUT_PATH]
               [--initial_path INITIAL_PATH] [--initial_epoch INITIAL_EPOCH]
               [--update_interval UPDATE_INTERVAL]
               [--learning_rate LEARNING_RATE] [--max_epochs MAX_EPOCHS]
               [--content_weight CONTENT_WEIGHT] [--style_weight STYLE_WEIGHT]
               [--temporal_weight TEMPORAL_WEIGHT]
               [--block_length BLOCK_LENGTH]
```

### Requirements

tensorflow, python3, scipy, numpy, Pillow, opencv
