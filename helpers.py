def get_tmap(controller):
    return controller.shared_data["tool_map"]

def get_pmap(controller):
    return controller.shared_data["pos"]

def april_to_position(controller_param, april_id):
        # get the mapping
        pos_map = get_pmap(controller_param)

        # look up position ID if it exists, otherwise fall back to raw
        return pos_map.get(april_id, april_id)

def position_to_april(controller_param, pos_id):

        # look up april ID if it exists, otherwise fall back to raw
        return next((k for k, v in get_pmap(controller_param).items() if v == pos_id), pos_id)

# Equation: width = height * aspect_ratio
def resize_to_fit_4_3(image, max_w, max_h):
    target_aspect = 4 / 3
    new_w = max_w
    new_h = int(max_w / target_aspect)
    if new_h > max_h:
        new_h = max_h
        new_w = int(max_h * target_aspect)
    if new_w > max_w:
        new_w = max_w
        new_h = int(max_w / target_aspect)
    return image.resize((new_w, new_h))

# VISION
