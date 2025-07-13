class ToolAdapter:
    def __init__(self, available_tools):
        # available_tools is a dict (tool_map)
        self.attached = {}
        self.available = available_tools

    def attach_tool(self, chosen_id, pose):
        # tool_id is an integer, and pose has six coords
        # chosen_id is ths id of the tool in the self.available dict
        tool_id = len(self.attached)
        self.attached[tool_id] = {
            "pose": pose,
            "name": self.available[chosen_id]
        }

        # Attaching Tool Code Here
        # attach(target_pos)

        print(f"<attach_tool> Tool '{tool_id}' attached at pose {pose}")
        return pose

    def detach_tool(self, tool_id, target_pos):
        if tool_id in self.attached:
            del self.attached[tool_id]
            
            # Detaching Code here
            # detach(target_pos)

            print(f"<detach_tool> Tool '{tool_id}' detached")
        else:
            print(f"<detach_tool> Tool '{tool_id}' not found")

    def move_tool_to(self, tool_id, pose, record):
        if tool_id in self.attached:
            self.attached[tool_id] = pose

            # Moving Tool Code here

            print(f"<move_tool_to> Tool '{tool_id}' moved to pose {pose}")
        else:
            print(f"<move_tool_to> Tool '{tool_id}' not attached")

    # def get_tool_status(self, tool_id):
    #     pose = self.attached[tool_id]["pose"]
    #     name = self.attached[tool_id]["name"]
    #     if pose:
    #         print(f"<get_tool_status> Tool '{name}' ({tool_id}) is at pose {pose}")
    #     else:
    #         print(f"<get_tool_status> Tool '{tool_id}' not attached")
    #     return pose