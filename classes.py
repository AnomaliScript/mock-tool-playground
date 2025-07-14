import math
import numpy as np

class Slots: 
    def __init__(self, center_pos, num_slots, radius=0.25):
        self.pos = center_pos
        self.limit = num_slots
        self.radius = radius
        # This dict will be filled with positions in the "register" member method
        self.slot_positions = {}

    def register(self):
        # Generate evenly spaced slot poses in a flat circle around center_pos.
        angle_step = 2 * math.pi / self.limit

        for i in range(self.limit):
            angle = i * angle_step
            x = self.pos[0] + self.radius * math.cos(angle)
            y = self.pos[1] + self.radius * math.sin(angle)
            z = self.pos[2]

            # Fake orientation for now (roll, pitch, yaw) all zero
            pose = np.array([x, y, z, 0.0, 0.0, 0.0])
            self.slot_positions[i] = pose
    
    # math
    def find_closest_slot(self, input_pose, threshold=0.01):
        best_match = None
        best_dist = float("inf")
        for i, slot_pose in self.slot_positions.items():
            dist = np.linalg.norm(input_pose[:3] - slot_pose[:3])
            if dist < best_dist and dist <= threshold:
                best_match = i
                best_dist = dist
        return best_match

# Slots.slot_positions will directly feed into ToolAdapter.p_positions

class ToolAdapter:
    def __init__(self, available_tools, holding_limit, possible_positions):
        # available_tools is a dict (tool_map)
        self.attached = {
            # For each tool_id:
            # "pose": {1, 2, 3, 4, 5, 6}
            # "slot" : slots.slot_positions[int]
            # "name": "scalpel"
        }
        self.available = available_tools
        self.limit = holding_limit
        # self.p_positions = {
        #     
        # }

    def attach_tool(self, chosen_id, pose, target_pos, slots_obj):
        # tool_id is an integer, and pose has six coords
        # chosen_id is ths id of the tool in the self.available dict
        # Has to return a string if there is an error
        if len(self.attached) >= self.limit:
            return "Max limit reached."
        tool_id = len(self.attached)
        
        slot_id = slots_obj.find_closest_slot(pose)
        self.attached[tool_id] = {
            "pose": pose,
            "slot": slot_id,
            "name": self.available[chosen_id]
        }

        # Attaching Tool Code Here
        # attach(target_pos)

        print(f"<attach_tool> Tool '{tool_id}' attached at slot {self.attached[tool_id]["pose"]}")
        return pose

    def detach_tool(self, tool_id, target_pos):
        if tool_id in self.attached:
            del self.attached[tool_id]
            
            # Detaching Code here
            # detach(target_pos)

            print(f"<detach_tool> Tool '{tool_id}' detached")
        else:
            print(f"<detach_tool> Tool '{tool_id}' not found")

    def move_tool_to(self, tool_id, pose, slots_obj):
        if tool_id in self.attached:
            slot_id = slots_obj.find_closest_slot(pose)

            # Moving Tool Code here
            
            print(f"<move_tool_to> Tool '{tool_id}' moved to slot {slot_id}")
        else:
            print(f"<move_tool_to> Tool '{tool_id}' not attached")

    # Depreciated probably
    # def get_tool_status(self, tool_id):
    #     pose = self.attached[tool_id]["pose"]
    #     name = self.attached[tool_id]["name"]
    #     if pose:
    #         print(f"<get_tool_status> Tool '{name}' ({tool_id}) is at pose {pose}")
    #     else:
    #         print(f"<get_tool_status> Tool '{tool_id}' not attached")
    #     return pose