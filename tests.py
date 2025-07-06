import torch
action = {}
main_head_actions = torch.tensor([0, 1, 2, 3, 4, 5])
action["inventory"] = main_head_actions == 0
action["camera_enabled"] = action["inventory"].logical_not() * ((main_head_actions - 1) % 2 == 1)

action["main_head"] = (main_head_actions - action["inventory"].logical_not().int()) // 2  # remove the inventory offset and the camera choice
print(action["main_head"])
print(action["camera_enabled"])
print(action["inventory"])
print((action["main_head"] * 2 + action["camera_enabled"] + 1) * action["inventory"].logical_not())