import pathlib

# general
class_num = 10
img_size = (16, 16)
hidden_layer_num = 1
weights_folder = "./weights/"
weights_file_prefix = f"{pathlib.Path(__file__).parent.resolve().__str__()}" \
                      f"/{weights_folder}usps-weights-h{hidden_layer_num}-class"

# train
epochs = 10
train_index = 6000
test_index_start = 7000
test_index_end = 8000
train = False
read = True

# simplify
round_num = 0

# preview
preview_window_size = 600
preview_font_scale = 2
preview_font_thickness = 4
default_img_name = "img.jpg"
