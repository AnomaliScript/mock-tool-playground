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