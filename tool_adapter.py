class ToolAdapter:
    def __init__(self):
        self.attached_tools = {}

    def attach_tool(self, tool_id, pose):
        self.attached_tools[tool_id] = pose
        # Attaching Tool Code Here
        print(f"[attach_tool] Tool '{tool_id}' attached at pose {pose}")

    def detach_tool(self, tool_id):
        if tool_id in self.attached_tools:
            del self.attached_tools[tool_id]
            # Detaching Code here
            print(f"[detach_tool] Tool '{tool_id}' detached")
        else:
            print(f"[detach_tool] Tool '{tool_id}' not found")

    def move_tool_to(self, tool_id, pose):
        if tool_id in self.attached_tools:
            self.attached_tools[tool_id] = pose
            # Moving Tool Code here
            print(f"[move_tool_to] Tool '{tool_id}' moved to pose {pose}")
        else:
            print(f"[move_tool_to] Tool '{tool_id}' not attached")

    def get_tool_status(self, tool_id):
        pose = self.attached_tools.get(tool_id)
        if pose:
            print(f"[get_tool_status] Tool '{tool_id}' is at pose {pose}")
        else:
            print(f"[get_tool_status] Tool '{tool_id}' not attached")
        return pose
    
adapter = ToolAdapter()
adapter.attach_tool("scalpel", (0.1, 0.2, 0.3, 0, 90, 0))
adapter.move_tool_to("scalpel", (0.15, 0.25, 0.3, 0, 90, 0))
adapter.get_tool_status("scalpel")
adapter.detach_tool("scalpel")