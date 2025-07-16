def is_int(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        return None
    
def get_tool_id(verb):
    while True:
        tool_id = input(f"Which tool would you like to {verb}? ")
        int_tool_id = is_int(tool_id)
        if int_tool_id is not None:
            return int_tool_id
        else:
            print("Please type the number corresponding to the tool.")